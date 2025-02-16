'''
####################################################################################################################
    P1 part 1
####################################################################################################################
'''
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal

N = 10000


def dynamics(Z):
    Z1, Z2 = Z[:, 0], Z[:, 1]
    dZ1 = np.tanh(Z1 ** 3)
    dZ2 = np.tanh(Z2 ** 3)
    return np.stack((dZ1, dZ2), axis=1)


t = 0
dt = 0.01
Z = np.random.normal(0, 1, size=(N, 2))
while t < 0.5:
    Z += dt * dynamics(Z)
    t += dt

np.random.seed(42)

plt.figure(figsize=(8, 6))
plt.hist2d(Z[:, 0], Z[:, 1], bins=50, density=True)
plt.colorbar(label='Density')
plt.title('2D Histogram of Transformed Distribution at t=0.5')
plt.xlabel('$Z_1(t)$')
plt.ylabel('$Z_2(t)$')
plt.grid(True)
plt.show()

''' 
####################################################################################################################
    P1 part 3 
####################################################################################################################
'''


def f(z):
    z1, z2 = z[:, 0], z[:, 1]
    f1 = np.tanh(z1 ** 3)
    f2 = np.tanh(z2 ** 3)
    return np.column_stack((f1, f2))


def jacobian_f(z):
    z1, z2 = z[:, 0], z[:, 1]
    d_f1_dz1 = 3 * z1 ** 2 * (1 - np.tanh(z1 ** 3) ** 2)
    d_f2_dz2 = 3 * z2 ** 2 * (1 - np.tanh(z2 ** 3) ** 2)

    jacobian = np.zeros((z.shape[0], 2, 2))
    jacobian[:, 0, 0] = d_f1_dz1
    jacobian[:, 1, 1] = d_f2_dz2
    return jacobian


def compute_log_prob_dynamics(log_p, z, dt):
    jacobian = jacobian_f(z)
    trace_term = -np.trace(jacobian, axis1=1, axis2=2)
    return log_p + (trace_term * dt)


def compute_density():
    mean = np.zeros(2)
    cov = np.eye(2)
    p_t = multivariate_normal.pdf(z_grid, mean=mean, cov=cov)

    densities = [p_t]
    dt = time_points[1] - time_points[0]
    for t in time_points[1:]:
        log_p_t = np.log(densities[-1])
        log_p_t = np.array(
            [compute_log_prob_dynamics(log_p, z.reshape(1, -1), dt)[0] for log_p, z in zip(log_p_t, z_grid)])
        p_t = np.exp(log_p_t)
        p_t /= np.sum(p_t)
        densities.append(p_t)
    return densities


time_points = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5])
z_min, z_max = -4, 4
x = np.linspace(z_min, z_max, 100)
y = np.linspace(z_min, z_max, 100)
z_grid = np.array([[xi, yi] for xi in x for yi in y])

densities = compute_density()

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()

xv, yv = np.meshgrid(x, y)
heatmap = None
for i, (t, density) in enumerate(zip(time_points, densities)):
    ax = axes[i]
    density_grid = density.reshape(100, 100)

    heatmap = ax.imshow(
        density_grid, extent=[z_min, z_max, z_min, z_max], origin='lower', cmap='hot'
    )
    ax.set_title(f"$t = {t:.1f}$")
    ax.set_xlim([z_min, z_max])
    ax.set_ylim([z_min, z_max])

plt.tight_layout()
plt.colorbar(heatmap, ax=axes.tolist(), orientation='vertical', shrink=0.8)
plt.show()
