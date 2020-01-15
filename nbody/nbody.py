import numpy as np
import scipy.integrate as integ
import scipy.optimize as opt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
_ = np.newaxis
plt.style.use("seaborn-paper")

# Intial conditions generator

def init_cond(n, b = 0):
    # Initial condition array as (pos, vel)
    # Sun, earth and mars
    G = 6.67408e-11
    if n == 1:
        N = 3
        m_sun, m_earth, m_mars = 1.989e30, 5.972e24, 6.39e23
        d_earth, d_mars = 149.6e9, 227.94e9
        T_earth = 365 * 24 * 3600
        T_mars = 1.881 * T_earth
        v_earth, v_mars = 2*np.pi*d_earth/T_earth, 2*np.pi*d_mars/T_mars
        r = np.zeros(6*N)
        r[3], r[6] = d_earth, d_mars
        r[13], r[16] = v_earth, v_mars
        return r, np.array([m_sun, m_earth, m_mars]), N

    if n == 11:
        N = 3
        m_sun, m_earth, m_mars, m_moon = 1.989e30, 5.972e24, 6.39e23, 7.3477e22
        d_e, d_mars = 149.6e9, 227.94e9
        T_earth = 365 * 24 * 3600
        T_mars = 1.881 * T_earth
        v_e, v_mars = 2*np.pi*d_e/T_earth, 2*np.pi*d_mars/T_mars
        x_earth = np.array([d_e, 0, 0])
        v_earth = np.array([0, v_e, 0])
        x_sun = np.zeros(3)
        v_sun = np.zeros(3)
        x_moon = np.array([d_e*(1 + .00256), 0, 0])
        v_moon = np.array([0, v_e + np.sqrt(G*(m_earth + m_moon)/d_e/.00256), 0])
        r = np.concatenate((x_sun, x_earth, x_moon, v_sun, v_earth, v_moon))
        return r, np.array([m_sun, m_earth, m_moon]), N

    elif n == 2:
        N = 8
        m_sun = 1.989e30
        m_jup = 1.89813e27
        m_sat = 5.683e26
        m_earth = 5.972e24
        m_mars = 6.39e23
        m_merc = 3.285e23
        m_merc = 3.285e23
        m_venus = 4.867e24
        m_uranus = 8.681e25

        r_sun = np.array([-3.898661230717496E-03, 7.415302864203051E-03, 2.578901732695751E-05])
        v_sun = np.array([-8.321319722382462E-06, -2.112060396506490E-06, 2.294436703343205E-07])

        r_jup = np.array([6.112580326089336E-01, -5.179429767804448E+00, 7.806108715270889E-03])
        v_jup = np.array([7.401556691279816E-03, 1.244293071313719E-03, -1.707393420780204E-04])

        r_sat = np.array([3.851623734693806E+00, -9.255327062504390E+00, 7.600262363748847E-03])
        v_sat = np.array([4.842287709206108E-03, 2.127139026783579E-03, -2.297713882243470E-04])

        r_earth = np.array([-3.717838810092671E-01, 9.194376380338656E-01, -1.700490195377209E-05])
        v_earth = np.array([-1.623980320236148E-02, -6.497278193338332E-03, 6.942121787282369E-08])

        r_mars = np.array([-1.217281139281842E+00, -9.984593235047629E-01, 8.719313130589665E-03])
        v_mars = np.array([9.447320450053694E-03, -9.577276515412677E-03, -4.324112821754592E-04])

        r_merc = np.array([1.934032082452039E-01, -3.834888578002187E-01, -5.001614824904102E-02])
        v_merc = np.array([1.947962864596816E-02, 1.407690977288430E-02, -6.370388074239325E-04])

        r_ve
        nus = np.array([6.611520265276862E-01, 2.935831802357576E-01, -3.442573108553538E-02])
        v_venus = np.array([-8.068856781239711E-03, 1.848890935302577E-02, 7.191031775268186E-04])

        r_uranus = np.array([1.619419144433039E+01, 1.142287523320143E+01, -1.673731301415755E-01])
        v_uranus = np.array([-2.296090233598074E-03, 3.030740520197179E-03, 4.104383183848175E-05])

        r = np.concatenate((r_sun, r_jup, r_sat, r_earth, r_mars, r_merc, r_venus, r_uranus, v_sun, v_jup, v_sat, v_earth, v_mars, v_merc, v_venus, v_uranus))
        return r, np.array([m_sun, m_jup, m_sat, m_earth, m_mars, m_merc, m_venus, m_uranus]), N

    elif n == 3:
        N = 3
        m0 = 1
        m1 = .001
        m2 = 1e-12
        x0 = np.array([0, 0, 0])
        x1 = np.array([-1, 0, 0]
        )
        x2 = np.array([b, -10, 0])
        v0 = np.array([0, 0, 0])
        v1 = np.array([0, -1.5, 0])
        v2 = np.array([0, 3, 0])
        return np.concatenate((x0, x1, x2, v0, v1, v2)), np.array([m0, m1, m2]), N

    elif n == 0:
        N = 3
        m_sun, m_earth, m_orb = 1.989e30, 5.972e24, 1e3
        d_e = 149.6e9
#        l1 = d_e * (m_earth / 3 / m_sun)**(1/3)
        l1 = opt.root_scalar(lambda r: get_l1(r, m_sun, m_earth, d_e), bracket=(.1, d_e-.1), xtol=1e-18)
        d_orb = d_e - l1.root
        T_earth = 365 * 24 * 3600
        T_orb = T_earth
        v_e, v_orb = 2*np.pi*d_e/T_earth, 2*np.pi*d_orb/T_orb
        x_earth = np.array([d_e, 0, 0])
        v_earth = np.array([0, v_e, 0])
        x_sun = np.zeros(3)
        v_sun = np.zeros(3)
#        x_orb = np.array([d_e/2*(1 + b), -d_e*np.sqrt(3)/2*(1 + b), 0])
#        v_orb = np.array([v_e*np.sqrt(3)/2, v_e/2, 0])
        x_orb = np.array([d_orb*(1+b), 0, 1e3])
        v_orb = np.array([0, v_orb, 0])
        r = np.concatenate((x_sun, x_earth, x_orb, v_sun, v_earth, v_orb))
        return r, np.array([m_sun, m_earth, m_orb]), N

    elif n == 10:
        N = 4
        m_sun, m_earth, m_orb, m_jupiter = 1.989e30, 5.972e24, 1e3, 1.898e27
        d_e = 149.6e9
        d_jup = 5.2*d_e
#        l1 = d_e * (m_earth / 3 / m_sun)**(1/3)
        l1 = opt.root_scalar(lambda r: get_l1(r, m_sun, m_earth, d_e), bracket=(.1, d_e-.1), xtol=1e-18)
        d_orb = d_e - l1.root
        T_earth = 365 * 24 * 3600
        T_orb = T_earth
        T_jup = T_earth*11.86
        v_e, v_orb = 2*np.pi*d_e/T_earth, 2*np.pi*d_orb/T_orb
        v_jup = 2*np.pi*d_jup/T_jup
        x_earth = np.array([d_e, 0, 0])
        v_earth = np.array([0, v_e, 0])
        x_sun = np.zeros(3)
        v_sun = np.zeros(3)
        x_orb = np.array([d_e/2*(1 + b), -d_e*np.sqrt(3)/2*(1 + b), 0])
        v_orb = np.array([v_e*np.sqrt(3)/2, v_e/2, 0])
#        x_orb = np.array([d_orb*(1+b), 0, 1e3])
#        v_orb = np.array([0, v_orb, 0])
        x_jup = np.array([d_jup, 0, 0])
        v_jup = np.array([0, v_jup, 0])
        r = np.concatenate((x_sun, x_earth, x_orb, x_jup, v_sun, v_earth, v_orb, v_jup))
        return r, np.array([m_sun, m_earth, m_orb, m_jupiter]), N


# Numerical calculator for l1

def get_l1(r, M1, M2, R):
    return M2/r**2 + M1/R**2 - r*(M1 + M2)/R**3 - M1/(R-r)**2

# Initial condition transformation

def cond_cm(r, m, N):
    x, v = r[:3*N], r[3*N:]
    xx = x.reshape(N, 3)
    x_cm = m[_, :].dot(xx) / m.sum()
    xx = xx - x_cm
    r[:3*N] = xx.flatten()
    vv = v.reshape(N, 3)
    v_cm = m[_,:].dot(vv) / m.sum()
    vv = vv - v_cm
    r[3*N:] = vv.flatten()
    return r

def normalize(r, m, N, G = 6.67408e-11):
    X = np.max(np.absolute(r[:3*N]))
    M = m.sum()
    V = np.sqrt(G * M / 2 / X)
    T = np.sqrt(9*X**3/(G*M))
    norm_cond = np.array([X, M, V, T])
    r[:3*N] /= X
    r[3*N:] /= V
    return r, m/M, norm_cond

# Integrator funtion

def g(t, r, m, N):
    x = r[:3*N]
    v = r[3*N:]
    xx = x.reshape(N, 3)
    d = xx[_, :, :] - xx[:, _, :]
    dist = np.sum(d**2, axis = -1)
    np.fill_diagonal(dist, 1)
    f = m[_, :, _] * d / (dist[:, :, _]**(3/2) + 0)
    f = np.sum(f, axis=1)
    return np.concatenate((2*v, 4*f.flatten()))

#for b in [1e-3]:
#    r0, m, N = init_cond(0, b = b)
#    r0 = cond_cm(r0, m, N)
#    r0, m, nc = normalize(r0, m, N)
#    ti, tf = 0, 4
#    sol = integ.solve_ivp(lambda t, r: g(t, r, m, N), [ti, tf], r0, max_step=.0001)
#    d = (sol.y[6]**2 + sol.y[7]**2) / (sol.y[3]**2 + sol.y[4]**2)
#    print(sol.status)
#    print(sol.message)
#    labels=["Sun", "Earth", "Satellite"]
#    for i in range(N):
#        if i == 0:
#            plt.plot(sol.y[i*3], sol.y[i*3+1], "o", label=labels[i])
#        else:
#            plt.plot(sol.y[i*3], sol.y[i*3+1], label=labels[i])
#    plt.legend()
#    #plt.xlim(-.2, .9)
#    #plt.ylim(-1.1, .2)
#    plt.savefig("../Article/figs/l5_orb.pdf", dpi=1000, transparent=True, bbox_inches="tight")
#    plt.show()
#    plt.plot(sol.t, d, label="Radial perturbation = {}".format(b))
#plt.legend()
#plt.xlabel("t (normalized)")
#plt.ylabel(r"1 - $\left( \frac{r}{R} \right)^2$")
##plt.xlim(0, 3.5)
#plt.savefig("../Article/figs/l5_radius_perturbation.pdf", dpi=1000, transparent=True, bbox_inches="tight")
#plt.show()



#sol = integ.solve_ivp(lambda t, r: np.concatenate((2*r[r.size//2:], 4*(np.sum( m[_, :, _] * (r[:r.size//2].reshape(r.size//6, 3)[_, :, :] - r[:r.size//2].reshape(r.size//6, 3)[:, _, :])/ (np.sum((r[:r.size//2].reshape(r.size//6, 3)[_, :, :] - r[:r.size//2].reshape(r.size//6, 3)[:, _, :] )**2, axis = -1) + np.diag(np.ones(r.size//6)))[:, :, _]**(3/2), axis=1)).flatten())), [0, 4], r0, max_step=.001)

b = 0
r0, m, N = init_cond(10, b = b)
r0 = cond_cm(r0, m, N)
r0, m, nc = normalize(r0, m, N)
ti, tf = 0, 1
sol = integ.solve_ivp(lambda t, r: g(t, r, m, N), [ti, tf], r0, max_step=.0001)
d = (sol.y[6]**2 + sol.y[7]**2) / (sol.y[3]**2 + sol.y[4]**2)
print(sol.status)
print(sol.message)
labels=["Sun", "Earth", "Satellite", "Jupiter"]
for i in range(N):
    if i == 0:
        plt.plot(sol.y[i*3], sol.y[i*3+1], "o", label=labels[i])
    else:
        plt.plot(sol.y[i*3], sol.y[i*3+1], label=labels[i])
plt.legend()
#plt.xlim(-.2, .9)
#plt.ylim(-1.1, .2)
#plt.savefig("../Article/figs/l5_orb.pdf", dpi=1000, transparent=True, bbox_inches="tight")
plt.show()
plt.plot(sol.t, d)
plt.xlabel("t (normalized)")
plt.ylabel(r"$1 - \left( \frac{r}{R} \right)^2$")
#plt.savefig("../Article/figs/l5_d_jup.pdf", dpi=1000, transparent=True, bbox_inches="tight")
plt.show()

#fig, axs = plt.subplots(2, 2, sharex='col', sharey='row',
#                        gridspec_kw={'hspace': 0, 'wspace': 0})

#(ax1, ax2), (ax3, ax4) = axs

#for t, ax in zip([1000, 6000, 12000, 18000], [ax1, ax2, ax3, ax4]):
#    for i in range(N):
#        if i == 0:
#            ax.plot(sol.y[i*3, :t], sol.y[i*3+1, :t], linestyle="--", label = labels[i])
#            ax.plot(sol.y[i*3, t], sol.y[i*3+1, t], "o", label = labels[i])
#        else:
#            ax.plot(sol.y[i*3, :t], sol.y[i*3+1, :t], linestyle="--", label = labels[i])
#            ax.plot(sol.y[i*3, t], sol.y[i*3+1, t], "o", label = labels[i])
#        ax.set_xlim(-1.2, 1.2)
#        ax.set_ylim(-1.2, 1.2)

#for ax in axs.flat:
#    ax.label_outer()

#ax1.legend()

#plt.savefig("../Article/figs/l5_pert_orbit.pdf", dpi=1000, transparent=True, bbox_inches="tight")
#plt.show()

#plt.plot(sol.t, d)
#plt.show()


def get_T(r, m, N):
    v = r[:3*N]
    v = v.reshape(N, 3)
    return np.sum(m * np.sum(v**2, axis = 1))

def get_V(r, m, N):
    x = r[:3*N]
    x = x.reshape(N, 3)
    d = x[_, :, :] - x[:, _, :]
    dist = np.sum(d**2, axis = -1)
    np.fill_diagonal(dist, 1)
    V = - m[:, _] * m[_, :] / np.sqrt(dist)/2
    np.fill_diagonal(V, 0)
    return V.sum()


#sol1 = integ.solve_ivp(lambda t, r: g(t, r, m, N), [ti, tf], r0, max_step=.001)
#sol2 = integ.solve_ivp(lambda t, r: g(t, r, m, N), [ti, tf], r0, max_step=.01)
#sol3 = integ.solve_ivp(lambda t, r: g(t, r, m, N), [ti, tf], r0, max_step=.001)

#E1 = np.zeros(sol1.t.size)
#T1 = np.zeros(sol1.t.size)
#V1 = np.zeros(sol1.t.size)
#E2 = np.zeros(sol2.t.size)
#T2 = np.zeros(sol2.t.size)
#V2 = np.zeros(sol2.t.size)
#E3 = np.zeros(sol3.t.size)
#T3 = np.zeros(sol3.t.size)
#V3 = np.zeros(sol3.t.size)


#for i in range(E1.size):
#    r = sol1.y[:, i]
#    V1[i] = get_V(r, m, N)
#    T1[i] = get_T(r, m, N)
#    E1[i] = get_T(r, m, N) + get_V(r, m, N)
#for i in range(E2.size):
#    r = sol2.y[:, i]
#    V2[i] = get_V(r, m, N)
#    T2[i] = get_T(r, m, N)
#    E2[i] = get_T(r, m, N) + get_V(r, m, N)
#for i in range(E3.size):
#    r = sol3.y[:, i]
#    V3[i] = get_V(r, m, N)
#    T3[i] = get_T(r, m, N)
#    E3[i] = get_T(r, m, N) + get_V(r, m, N)

#print(-T3[-1]/V3[-1])

#plt.plot(sol1.t, E1)
#plt.plot(sol2.t, E2, label="Maximum step: 0.05")
#plt.plot(sol3.t, E3, label="Maximum step: 0.01")
#plt.xlabel("t (normalized)")
#plt.ylabel("Total energy (arbitrary units)")
#plt.legend()
#plt.savefig("../Article/figs/sun_earth_moon_energy.pdf", dpi=1000, transparent=True, bbox_inches="tight")
#plt.show()

#x0 = np.array([2, 0, 0])
#v0 = np.array([0, 2, 0])
#m0 = 1e-8
#r0, m, N = add_probe_ic(r0, m, x0, v0, m0, N)
