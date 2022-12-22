from meas import *

N = 100
V_lim = [2.3, 2.6, -2.4, -2.9]
ss = 1.2

def pulse_s(G, alfa, beta):
    """
    Get set pulse number from conductance G.
    """
    G_inf = 1/(1-np.exp(-beta*N**alfa))
    return (-np.log(1-G/G_inf)/beta)**(1/alfa)

def pulse_r(G, alfa, beta):
    """
    Get reset pulse number from conductance G.
    """
    G_inf = 1-1/(1-np.exp(-beta*N**alfa))
    return N -(-np.log(1-(1-G)/(1-G_inf))/beta)**(1/alfa)

def pulse_n(G_c, G_n, popt):
    """
    Get delta pulse number from current and new conductance values.
    """
    G_i = np.clip(G_c, 0, 1)
    G_f = np.clip(G_n, 0, 1)
    if G_f > G_i:
        return pulse_s(G_f, *popt['s']) - pulse_s(G_i, *popt['s'])
    else:
        return pulse_r(G_f, *popt['r']) - pulse_r(G_i, *popt['r'])

def characterize(n):
    """
    Retrieves pulses / resistance characteristic with AWG - OSC.
    """
    V = [*np.linspace(V_lim[0], V_lim[1], N), *np.linspace(V_lim[2], V_lim[3], N)]
    wf_dict_AWG = {
        'V' : V, # pulse voltages
        'n' : [1 for i in range(N*2)],   # pulse repetitions
        'W' : [2e-4 for i in range(N*2)],     # pulse widths
        'T' : 2e-3,     # pulse period
        'read' : True,
        'read_V' : read_V
    }
    get_and_set_wf('AWG', wf_dict_AWG, ch_AWG)
    P = np.array([])
    P_i  = np.concatenate([np.arange(N), np.arange(-1, -N-1, -1)])
    R = []
    for i in range(n):
        P = np.concatenate([P, P_i])
        results_AWG = meas_AWG(wf_dict_AWG, [1, 2], R_s=1e3, R_min=5e3, plot = False)
        R.extend(results_AWG['R_mean'])
    return P, np.array(R)

def get_params(P, R, plot=True):
    """
    Retrieves set and reset functions' parameters from pulses / resistance characteristic returned by characterize.
    """
    points_s = np.where(np.array(P) > 0)[0]
    points_r = np.where(np.array(P) < 0)[0]
    y_s = P[points_s]
    y_r = P[points_r]+N
    x_s = 1/np.array(R[points_s])
    x_r = 1/np.array(R[points_r])
    A = min(x_s.min(), x_r.min())
    x_s /= A
    x_r /= A
    x_s -= 1
    x_r -= 1
    B = max(x_s.max(), x_r.max())
    x_s /= B
    x_r /= B
    x_s *= ss
    x_r *= ss
    x_s -= (ss-1)/2
    x_r -= (ss-1)/2
    
    points_s = np.where(np.logical_and(np.array(x_s) >= 0, np.array(x_s) <= 1))[0]
    points_r = np.where(np.logical_and(np.array(x_r) >= 0, np.array(x_r) <= 1))[0]
    popt_s, pcov_s = curve_fit(pulse_s, x_s[points_s], y_s[points_s], [1, 1e-2])
    popt_r, pcov_r = curve_fit(pulse_r, x_r[points_r], y_r[points_r], [1, 1e-2])
    x = np.linspace(0, 1, 101)
    if plot:
        plt.plot(x_s, y_s, '.b')
        plt.plot(x_r, y_r, '.r')
        plt.plot(x, pulse_s(x, *popt_s), '-b')
        plt.plot(x, pulse_r(x, *popt_r), '-r')
        plt.show()
    params = {
        'N' : N,
        'A' : A,
        'B' : B,
        'popt' : {
            's' : popt_s,
            'r' : popt_r
        }
    }
    return params 

def read(params):
    """
    Reads conductance value with AWG - OSC.
    """
    wf_dict = {
        'V' : [0], # pulse voltages
        'n' : [0],   # pulse repetitions
        'W' : [2e-4],     # pulse widths
        'T' : 2e-3,     # pulse period
        'read' : True,
        'read_V' : read_V
    }
    get_and_set_wf('AWG', wf_dict, ch_AWG)
    results = meas_AWG(wf_dict, [1, 2], R_s=1e3, R_min=5e3, plot=False)
    return (results['R_mean'][0]**-1/params['A'] - 1)/params['B'] * ss - (ss-1)/2


def read_(params):
    """
    Reads conductance value with SMU.
    """
    wf_dict = {
        'V' : [0],
        'n' : [0],
        'W' : [0],
        'T' : 1,
        'read' : True,
        'read_V' : read_V
    }
    results = meas_AWG_SMU([], [], wf_dict, R_s = 1e3, plot=False)
    return (results['R'][0]**-1/params['A'] - 1)/params['B'] * ss - (ss-1)/2

def write(params, G):
    """
    Writes conductance value G with AWG.
    """
    G_i = np.clip(read(params), 0, 1)
    G_f = np.clip(G, 0, 1)
    n = pulse_n(G_i, G_f, params['popt'])
    
    if n > 0:
        i0, i1, g0, g1 = 0, 1, G_i, G_f
    else:
        i0, i1, g0, g1 = 2, 3, 1-G_i, 1-G_f
    n = abs(round(n))
    dV = V_lim[i1] - V_lim[i0]
    V = np.linspace(dV*g0, dV*g1, n) + V_lim[i0]
    if n > 0:
        wf_dict_AWG = {
            #'V' : [2.4 if n>0 else -2.6], # pulse voltages
            #'n' : [round(abs(n))],   # pulse repetitions
            'V' : V, # pulse voltages
            'n' : [1 for i in range(n)],   # pulse repetitions
            'W' : [2e-4 for i in range(n)],     # pulse widths
            'T' : 2e-3,     # pulse period
            'read' : False,
            'read_V' : 0,
            'ch' : ch_AWG
        }
        get_and_set_wf('AWG', wf_dict_AWG, ch_AWG)
        DAQ = instruments['DAQ']
        AWG = instruments['AWG']
        DAQ.set_conn(ch_DAQ_AWG)
        AWG.set_outp(ch_AWG, 1)
        AWG.trigger()
        AWG.set_outp(ch_AWG, 0)
    