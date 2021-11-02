# Copyright (c) 2021, Tencent Inc. All rights reserved.

import taichi as ti
import numpy as np
from enum import Enum, auto

# Advection schemes
class AdvectionType(Enum):
  PIC = 0
  FLIP = 1
  NFLIP = 2
  SFLIP = 3
  APIC = 4
  AFLIP = 5
  ASFLIP = 6
  COUNT = 7


# Advection parameters
flip_velocity_adjustment = 0.0
flip_position_adjustment_min = 0.0
flip_position_adjustment_max = 0.0
apic_affine_stretching = 1.0
apic_affine_rotation = 1.0
particle_collision = 0.0

# Different advection scheme corresponds to different parameters.
# Please check our paper for more details.
def SetupAdvection(advection_type):
  global flip_velocity_adjustment
  global flip_position_adjustment_min, flip_position_adjustment_max
  global apic_affine_stretching, apic_affine_rotation
  global particle_collision
  if advection_type is AdvectionType.PIC:
    flip_velocity_adjustment = 0.0
    flip_position_adjustment_min = 0.0
    flip_position_adjustment_max = 0.0
    apic_affine_stretching = 0.0
    apic_affine_rotation = 0.0
    particle_collision = 0.0
  elif advection_type is AdvectionType.FLIP:
    flip_velocity_adjustment = 0.99
    flip_position_adjustment_min = 0.0
    flip_position_adjustment_max = 0.0
    apic_affine_stretching = 0.0
    apic_affine_rotation = 0.0
    particle_collision = 0.0
  elif advection_type is AdvectionType.NFLIP:
    flip_velocity_adjustment = 0.97
    flip_position_adjustment_min = 1.0
    flip_position_adjustment_max = 1.0
    apic_affine_stretching = 0.0
    apic_affine_rotation = 0.0
    particle_collision = 0.0
  elif advection_type is AdvectionType.SFLIP:
    flip_velocity_adjustment = 0.99
    flip_position_adjustment_min = 0.0
    flip_position_adjustment_max = 1.0
    apic_affine_stretching = 0.0
    apic_affine_rotation = 0.0
    particle_collision = 1.0
  elif advection_type is AdvectionType.APIC:
    flip_velocity_adjustment = 0.0
    flip_position_adjustment_min = 0.0
    flip_position_adjustment_max = 0.0
    apic_affine_stretching = 1.0
    apic_affine_rotation = 1.0
    particle_collision = 0.0
  elif advection_type is AdvectionType.AFLIP:
    flip_velocity_adjustment = 0.99
    flip_position_adjustment_min = 0.0
    flip_position_adjustment_max = 0.0
    apic_affine_stretching = 1.0
    apic_affine_rotation = 1.0
    particle_collision = 0.0
  elif advection_type is AdvectionType.ASFLIP:
    flip_velocity_adjustment = 0.99
    flip_position_adjustment_min = 0.0
    flip_position_adjustment_max = 1.0
    apic_affine_stretching = 1.0
    apic_affine_rotation = 1.0
    particle_collision = 1.0
  return advection_type


# Set current scheme
current_advection = SetupAdvection(AdvectionType.ASFLIP)

# Scheme label position
scheme_label_offset_x = -0.07

# Run Taichi on GPU
ti.init(arch=ti.gpu)
window_res = 512
paused = False

# A larger value can be used for higher-res simulations
quality = 1
n_grid = 96 * quality
dx, inv_dx = 1 / n_grid, float(n_grid)

# Particle source setting
init_particle_center_x = 0.5
init_particle_center_y = 0.15 + dx * 3.0
init_particle_size_x = 1.0 - dx * 6.0
init_particle_size_y = 0.3
n_particles = int(init_particle_size_x * init_particle_size_y * n_grid * n_grid * 9)

# dt setting
frame_dt = 4e-3
dt = 1e-4 / quality

# volume and mass
p_vol, p_rho = (dx * 0.5) ** 2, 1400
p_mass = p_vol * p_rho

# mechanics parameters
# Young's modulus and Poisson's ratio
E, nu = 5e5, 0.3
# Bulk modulus and shear modulus
kappa_0, mu_0 = E / (3 * (1 - nu * 2)), E / (2 * (1 + nu))
# plasticity parameters
friction_angle = 40.0
sin_phi = ti.sin(friction_angle / 180.0 * 3.141592653)
material_friction = 1.633 * sin_phi / (3.0 - sin_phi)
volume_recovery_rate = 0.5

# Collision object, here we use a simple rotating capsule for demo
init_capsule_center_x = 0.5
init_capsule_center_y = 0.6
init_capsule_vel_y = -1.0
capsule_move_frame = int((0.3 - init_capsule_center_y) / init_capsule_vel_y / frame_dt)
capsule_radius = 0.15
capsule_half_length = 0.05
capsule_rotation = ti.Vector.field(1, dtype=float, shape=())
capsule_angular_vel = 80.0
capsule_translation = ti.Vector.field(2, dtype=float, shape=())
capsule_trans_vel = ti.Vector.field(2, dtype=float, shape=())
capsule_friction = 1.0 - ti.exp(-0.4332 * dt / (dx * dx))

ground_friction = 1.0 - ti.exp(-0.1394 * dt / (dx * dx))
side_friction = 0.0

# Material points
x = ti.Vector.field(2, dtype=float, shape=n_particles)  # position
v = ti.Vector.field(2, dtype=float, shape=n_particles)  # velocity
C = ti.Matrix.field(2, 2, dtype=float, shape=n_particles)  # affine velocity field
F = ti.Matrix.field(2, 2, dtype=float, shape=n_particles)  # deformation gradient
Jp = ti.field(dtype=float, shape=n_particles)  # plastic deformation
grid_v = ti.Vector.field(
  2, dtype=float, shape=(n_grid, n_grid)
)  # grid node momentum/velocity
grid_v0 = ti.Vector.field(
  2, dtype=float, shape=(n_grid, n_grid)
)  # grid node previous velocity
grid_m = ti.field(dtype=float, shape=(n_grid, n_grid))  # grid node mass
gravity = ti.Vector.field(2, dtype=float, shape=())
adv_params = ti.Vector.field(6, dtype=float, shape=())

# Function converting world space coordinates to local (object) space
@ti.func
def WorldSpaceToMaterialSpace(x, translation, rotation):
  tmp = x - translation
  X = rotation.transpose() @ tmp
  return X


# Function computing the signed distance field of a 2D capsule
@ti.func
def SdfCapsule(X, radius, half_length):
  alpha = min(max((X[0] / half_length + 1.0) * 0.5, 0.0), 1.0)
  tmp = ti.Vector([X[0], X[1]])
  tmp[0] += (1.0 - 2.0 * alpha) * half_length
  return tmp.norm() - radius


# Function computing the gradient of signed distance field of a 2D capsule
@ti.func
def SdfNormalCapsule(X, radius, half_length):
  unclamped_alpha = (X[0] / half_length + 1.0) * 0.5
  alpha = min(max(unclamped_alpha, 0.0), 1.0)
  normal = ti.Vector([X[0], X[1]])
  normal[0] += (1.0 - 2.0 * alpha) * half_length
  ltmp = max(1e-12, normal.norm())
  normal[0] /= ltmp
  normal[1] /= ltmp
  if unclamped_alpha >= 0.0 and unclamped_alpha <= 1.0:
    normal[0] = 0.0
  return normal


# Project the singular values of deformation gradient with Drucker-Prager model
# Refer to [Yue et al. 2018] for details.
@ti.func
def ProjectDruckerPrager(S: ti.template(), Jp: ti.template()):
  JSe = S[0, 0] * S[1, 1]
  for d in ti.static(range(2)):
    S[d, d] = max(1e-6, abs(S[d, d] * Jp))

  if S[0, 0] * S[1, 1] >= 1.0:  # Project to tip
    S[0, 0] = 1.0
    S[1, 1] = 1.0
    Jp *= ti.pow(max(1e-6, JSe), volume_recovery_rate)
  else:  # Check if the stress is inside the feasible region
    Jp = 1.0
    Je = max(1e-6, S[0, 0] * S[1, 1])
    sqrS_0 = S[0, 0] * S[0, 0]
    sqrS_1 = S[1, 1] * S[1, 1]
    trace_b_2 = (sqrS_0 + sqrS_1) / 2.0
    Je2 = Je * Je
    yield_threshold = -material_friction * kappa_0 * 0.5 * (Je2 - 1.0)
    dev_b0 = sqrS_0 - trace_b_2
    dev_b1 = sqrS_1 - trace_b_2
    norm2_dev_b = dev_b0 * dev_b0 + dev_b1 * dev_b1
    mu_norm_dev_b_bar = mu_0 * ti.sqrt(norm2_dev_b / Je)

    if mu_norm_dev_b_bar > yield_threshold:  # Project to the yield surface
      det_b = sqrS_0 * sqrS_1
      det_dev_b = dev_b0 * dev_b1
      lambda_2 = yield_threshold / max(1e-6, mu_norm_dev_b_bar)
      lambda_1 = ti.sqrt(max(0.0, det_b - lambda_2 * lambda_2 * det_dev_b))
      S[0, 0] = ti.sqrt(abs(lambda_1 + lambda_2 * dev_b0))
      S[1, 1] = ti.sqrt(abs(lambda_1 + lambda_2 * dev_b1))


# Compute stress with Simo's [1982] neo-Hookean elasticity
@ti.func
def NeoHookeanElasticity(U, sig):
  J = sig[0, 0] * sig[1, 1]
  mu_J_1_2 = mu_0 * ti.sqrt(J)
  J_prime = kappa_0 * 0.5 * (J * J - 1.0)
  sqrS_1_2 = (sig[0, 0] * sig[0, 0] + sig[1, 1] * sig[1, 1]) / 2.0
  stress = ti.Matrix.identity(float, 2)
  stress[0, 0] = (sig[0, 0] * sig[0, 0] - sqrS_1_2) * mu_J_1_2
  stress[1, 1] = (sig[1, 1] * sig[1, 1] - sqrS_1_2) * mu_J_1_2
  stress = U @ stress @ U.transpose()
  stress[0, 0] += J_prime
  stress[1, 1] += J_prime
  return stress

# Check if a position is inside the capsule and compute their
# relative velocity and normal
@ti.func
def CheckSdfCapsule(pos, vel):
  cap_rot = capsule_rotation[None][0]
  capsule_rotmat = ti.Matrix(
    [
      [ti.cos(cap_rot), -ti.sin(cap_rot)],
      [ti.sin(cap_rot), ti.cos(cap_rot)],
    ]
  )
  local_pos = WorldSpaceToMaterialSpace(
    pos, capsule_translation[None], capsule_rotmat
  )
  phi = SdfCapsule(local_pos, capsule_radius, capsule_half_length)
  inside = False
  dotnv = 0.
  diff_vel = ti.Vector.zero(float, 2)
  n = ti.Vector.zero(float, 2)
  if phi < 0.0:
    n = capsule_rotmat @ SdfNormalCapsule(
      local_pos, capsule_radius, capsule_half_length
    )
    solid_vel = ti.Vector(
      [
        capsule_trans_vel[None][0]
        - capsule_angular_vel
        * (pos[1] - capsule_translation[None][1]),
        capsule_trans_vel[None][1]
        + capsule_angular_vel
        * (pos[0] - capsule_translation[None][0]),
      ]
    )
    diff_vel = solid_vel - vel
    dotnv = n.dot(diff_vel)
    if dotnv > 0.0:
      inside = True
  return inside, dotnv, diff_vel, n

# Sub-stepping the simulation
@ti.kernel
def Substep():
  # Advance the capsule
  capsule_rotation[None][0] += capsule_angular_vel * dt
  capsule_translation[None] += capsule_trans_vel[None] * dt

  for i, j in grid_m:
    grid_v[i, j] = [0, 0]
    grid_v0[i, j] = [0, 0]
    grid_m[i, j] = 0
  # Particle state update and scatter to grid (P2G)
  param_apic_str = adv_params[None][3]
  param_apic_rot = adv_params[None][4]
  rc0 = (param_apic_str + param_apic_rot) * 0.5
  rc1 = (param_apic_str - param_apic_rot) * 0.5
  for p in x:
    xp = x[p]
    vp = v[p]
    base = (xp * inv_dx - 0.5).cast(int)
    fx = xp * inv_dx - base.cast(float)
    # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
    w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
    # deformation gradient update
    Fp = F[p]
    Cp = C[p]
    Fp = (ti.Matrix.identity(float, 2) + dt * Cp) @ Fp
    U, sig, V = ti.svd(Fp)
    # Plasticity flow
    ProjectDruckerPrager(sig, Jp[p])
    # Reconstruct elastic deformation gradient after plasticity
    F[p] = U @ sig @ V.transpose()
    stress = NeoHookeanElasticity(U, sig)
    stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress
    affine_without_stress = p_mass * (Cp * rc0 + Cp.transpose() * rc1)
    affine = stress + affine_without_stress
    for i, j in ti.static(ti.ndrange(3, 3)):  # Loop over 3x3 grid node neighborhood
      offset = ti.Vector([i, j])
      dpos = (offset.cast(float) - fx) * dx
      weight = w[i][0] * w[j][1]
      grid_v[base + offset] += weight * (p_mass * vp + affine @ dpos)
      grid_v0[base + offset] += weight * (
        p_mass * vp + affine_without_stress @ dpos
      )
      grid_m[base + offset] += weight * p_mass
  # External force and collision
  for i, j in grid_m:
    nmass = grid_m[i, j]
    if nmass > 0:  # No need for epsilon here
      nvel = grid_v[i, j]
      nvel *= 1. / nmass  # Momentum to velocity
      nvel += dt * gravity[None]  # gravity
      grid_v0[i, j] *= 1. / nmass

      # Boundary conditions at border
      if i < 3 and nvel[0] < 0:
        nvel[0] = 0
        nvel[1] *= 1.0 - side_friction
      if i > n_grid - 3 and nvel[0] > 0:
        nvel[0] = 0
        nvel[1] *= 1.0 - side_friction
      if j < 3 and nvel[1] < 0:
        nvel[0] *= 1.0 - ground_friction
        nvel[1] = 0
      if j > n_grid - 3 and nvel[1] > 0:
        nvel[0] *= 1.0 - side_friction
        nvel[1] = 0
      # Boundary condition at capsule
      npos = ti.Vector([i, j]).cast(float) * dx
      inside, dotnv, diff_vel, n = CheckSdfCapsule(npos, nvel)
      if inside:
        dotnv_frac = dotnv * (1.0 - capsule_friction)
        nvel += diff_vel * capsule_friction + n * dotnv_frac
      grid_v[i, j] = nvel

  # grid to particle (G2P)
  param_flip_vel_adj = adv_params[None][0]
  param_flip_pos_adj_min = adv_params[None][1]
  param_flip_pos_adj_max = adv_params[None][2]
  param_part_col = adv_params[None][5] > 0.0
  for p in x:
    xp = x[p]
    base = (xp * inv_dx - 0.5).cast(int)
    fx = xp * inv_dx - base.cast(float)
    w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
    new_v = ti.Vector.zero(float, 2)
    new_C = ti.Matrix.zero(float, 2, 2)
    for i, j in ti.static(ti.ndrange(3, 3)):  # loop over 3x3 grid node neighborhood
      dpos = ti.Vector([i, j]).cast(float) - fx
      g_v = grid_v[base + ti.Vector([i, j])]
      weight = w[i][0] * w[j][1]
      new_v += weight * g_v
      new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)
    # Check if velocity adjustment is used (for any xFLIP)
    if param_flip_vel_adj > 0.0:
      vp = v[p]
      flip_pos_adj = param_flip_pos_adj_max
      # Check if our positional correction is adopted
      if flip_pos_adj > 0.0 and param_part_col:
        # Check if the particle collides with the capsule
        inside, _0, _1, _2 = CheckSdfCapsule(xp, vp)
        if inside:
            flip_pos_adj = 0.0
      # if not collided, check if the particle is separating
      if param_flip_pos_adj_min < flip_pos_adj:
        logdJ = new_C.trace() * dt
        J = F[p].determinant()
        if ti.log(max(1e-15, J)) + logdJ < -0.001:  # if not separating
          flip_pos_adj = param_flip_pos_adj_min
      # interpolate to get old nodal velocity
      old_v = ti.Vector.zero(float, 2)
      for i, j in ti.static(ti.ndrange(3, 3)):
        g_v0 = grid_v0[base + ti.Vector([i, j])]
        weight = w[i][0] * w[j][1]
        old_v += weight * g_v0
      # apply generalized FLIP advection
      diff_vel = vp - old_v
      v[p] = new_v + param_flip_vel_adj * diff_vel
      x[p] = xp + (new_v + flip_pos_adj * param_flip_vel_adj * diff_vel) * dt
    else:
      # apply PIC advection
      v[p] = new_v
      x[p] = xp + new_v * dt
    C[p] = new_C


# Function to reset the simulation
@ti.kernel
def Reset():
  for i in range(n_particles):
    x[i] = [
      (ti.random() - 0.5) * init_particle_size_x + init_particle_center_x,
      (ti.random() - 0.5) * init_particle_size_y + init_particle_center_y,
    ]
    v[i] = [0, 0]
    F[i] = ti.Matrix([[1, 0], [0, 1]])
    Jp[i] = 1
    C[i] = ti.Matrix.zero(float, 2, 2)
  gravity[None] = [0, -9.81]
  capsule_translation[None] = [init_capsule_center_x, init_capsule_center_y]
  capsule_trans_vel[None] = [0, init_capsule_vel_y]
  capsule_rotation[None] = [0.0]


print("[Hint] Press R to reset. <Space> to pause. <Left>/<Right> to switch schemes.")
gui = ti.GUI("ASFLIP Demo", res=window_res, background_color=0xFFFFFF)
Reset()
adv_params[None] = [
  flip_velocity_adjustment,
  flip_position_adjustment_min,
  flip_position_adjustment_max,
  apic_affine_stretching,
  apic_affine_rotation,
  particle_collision,
]

# Function to draw the capsule
def DrawCapsule(gui, radius, half_length, translation, rotation, color):
  phi = rotation.to_numpy()[0]
  ct = translation.to_numpy()
  psi = np.arctan2(radius, half_length)
  d = np.sqrt(radius * radius + half_length * half_length)
  vert = np.array(
    [
      [ct[0] + d * np.cos(phi + psi), ct[1] + d * np.sin(phi + psi)],
      [ct[0] - d * np.cos(phi - psi), ct[1] - d * np.sin(phi - psi)],
      [ct[0] - d * np.cos(phi + psi), ct[1] - d * np.sin(phi + psi)],
      [ct[0] + d * np.cos(phi - psi), ct[1] + d * np.sin(phi - psi)],
    ]
  )
  end_pos = np.array(
    [
      [ct[0] + half_length * np.cos(phi), ct[1] + half_length * np.sin(phi)],
      [ct[0] - half_length * np.cos(phi), ct[1] - half_length * np.sin(phi)],
    ]
  )
  gui.triangles(
    np.array([vert[0], vert[0]]),
    np.array([vert[1], vert[2]]),
    np.array([vert[2], vert[3]]),
    color=color,
  )
  gui.circles(end_pos, color=color, radius=radius * window_res)
  gui.text(current_advection.name, pos=[ct[0] + scheme_label_offset_x, ct[1]], font_size=30)


# Print scheme parameters
def PrintScheme():
  print("Advection Scheme: " + current_advection.name)
  print("FLIP Vel. Adj.: " + str(flip_velocity_adjustment))
  print("FLIP Pos. Adj. Min.: " + str(flip_position_adjustment_min))
  print("FLIP Pos. Adj. Max.: " + str(flip_position_adjustment_max))
  print("APIC Aff. Str.: " + str(apic_affine_stretching))
  print("APIC Aff. Rot.: " + str(apic_affine_rotation))
  print("Part. Col.: " + str(particle_collision))


PrintScheme()

# Begin the simulation
frame = 0
wid_frame = gui.label("Frame")
wid_frame.value = frame
while True:
  # Handle keyboard input
  if gui.get_event(ti.GUI.PRESS):
    if gui.event.key == "r":
      Reset()
      frame = 0
    elif gui.event.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
      break
    elif gui.event.key == " ":
      paused = not paused
    elif gui.event.key == ti.GUI.LEFT:
      if current_advection.value == 0:
        current_advection = AdvectionType(AdvectionType.COUNT.value - 1)
      else:
        current_advection = AdvectionType(current_advection.value - 1)
      SetupAdvection(current_advection)
      PrintScheme()
      adv_params[None] = [
        flip_velocity_adjustment,
        flip_position_adjustment_min,
        flip_position_adjustment_max,
        apic_affine_stretching,
        apic_affine_rotation,
        particle_collision,
      ]
    elif gui.event.key == ti.GUI.RIGHT:
      current_advection = AdvectionType(
        (current_advection.value + 1) % AdvectionType.COUNT.value
      )
      SetupAdvection(current_advection)
      PrintScheme()
      adv_params[None] = [
        flip_velocity_adjustment,
        flip_position_adjustment_min,
        flip_position_adjustment_max,
        apic_affine_stretching,
        apic_affine_rotation,
        particle_collision,
      ]
  # run simulation if not paused
  if not paused:
    for s in range(int(frame_dt // dt)):
      Substep()
    # if frame == 210: paused = True
    frame += 1
    wid_frame.value = frame
    if frame > capsule_move_frame:
      capsule_trans_vel[None] = [0, 0]
  # draw particles and UI
  gui.circles(x.to_numpy(), radius=1.5, color=0x068587)
  DrawCapsule(
    gui,
    capsule_radius,
    capsule_half_length,
    capsule_translation,
    capsule_rotation,
    0x035354,
  )
  gui.show()  # Change to gui.show(f'{frame:06d}.png') to write images to disk
