import matplotlib.pyplot as plt
import json
import os
from scipy.optimize import curve_fit
from instruments import *
from datetime import date

N = 100
ch_AWG, ch_SMU = 1, 2
ch_DAQ_AWG, ch_DAQ_SMU = 113, 111
read_V = .1
V_lim = [2.0, 2.6, -2.4, -3.0]
ss = 1.2

instruments = {}
rm = pyvisa.ResourceManager()
lr = rm.list_resources()
for r in lr:
    if "awg" in r:
        instruments['AWG'] = ArbitraryWaveformGenerator(r)
    elif "daq" in r:
        instruments['DAQ'] = DataAcquisition(r)
    #elif "172.31.182.32" in r:
    elif "smu" in r:
        instruments['SMU'] = SourceMeasureUnit(r)
    elif "osc" in r:
        instruments['OSC'] = Oscilloscope(r)

def connect(instruments):
    for key in instruments.keys():
        instruments[key].connect()

def disconnect(instruments):
    for key in instruments.keys():
        instruments[key].disconnect()
        
def reset(instruments):
    for key in instruments.keys():
        instruments[key].reset()

def my_wf(V, n, W, T, read, read_V):
    SRAT = 10/np.array(W).min() if np.array(W).min() > 0 else 1
    W_len = np.round(np.array(W)*SRAT).astype(int)
    T_len = round(T*SRAT)
    if not len(V) == len(n) == len(W):
        raise RuntimeError("Inputs V, n, and W of my_wf have different lengths.")
    for i in range(len(V)):
        if i == 0:
            n_read = T_len-W_len[i]
            n_zeros = T_len//2
            wf = [0 for j in range(n_zeros)] + [(read_V if read else 0) for j in range(n_read)]
        for j in range(n[i]):
            wf += [V[i] for k in range(W_len[i])]
            wf += [(read_V if read else 0) for k in range(T_len - W_len[i])]
    wf += [0 for j in range(n_zeros)]
    return wf, SRAT

def get_and_set_wf(key, wf_dict, aint=False):
    inst = instruments[key]
    wf, SRAT = my_wf(wf_dict['V'], wf_dict['n'], wf_dict['W'], wf_dict['T'], wf_dict['read'], wf_dict['read_V'])
    if aint:
        inst.set_wf(wf, SRAT, wf_dict['ch'], aint=aint)
    else:
        inst.set_wf(wf, SRAT, wf_dict['ch'])

def get_valid(wf_i, T_len, W_len, read_V, tol=.05, wf_o=[]):
    thr = 1e-4*read_V
    a = np.zeros(W_len)
    b = np.ones(T_len//5)/(T_len//5)
    w = np.concatenate([a, b])
    temp = np.convolve(wf_i, w, "same")
    valid = np.logical_and(read_V * (1-tol) < temp, temp < read_V * (1+tol))
    temp = abs(np.gradient(wf_i)) < thr
    valid = np.logical_and(valid, temp)
    if len(wf_o):
        temp = abs(np.gradient(np.convolve(wf_o, w, "same"))) < thr
        valid = np.logical_and(valid, temp)
    return valid

def get_R_mean_(R, valid, W_len):
    valid_idxs = np.where(valid)[0]
    R_a_b = []
    R_i = [R[valid_idxs[0]]]
    start = valid_idxs[0]
    prev_idx = valid_idxs[0]
    for idx in valid_idxs[1:]:
        if idx - prev_idx > W_len:
            R_a_b.append((np.mean(R_i), start, prev_idx))
            R_i = [R[idx]]
            start = idx
        else:
            R_i.append(R[idx])
        prev_idx = idx
    R_a_b.append((np.mean(R_i), start, prev_idx))
    R_mean = np.array(R_a_b)[:, 0]
    a = np.array(R_a_b)[:, 1].astype(int)
    b = np.array(R_a_b)[:, 2].astype(int)
    return R_mean, a, b

def get_R_mean(t, V_i, read_V, R, valid, T_len, W_len, n):
    start = np.where(V_i > read_V*.9)[0][0]
    #start = np.argmin(abs(t))
    a = np.arange(start, start+T_len*n, T_len)
    b = np.arange(start+T_len, start+T_len*(n+1)-W_len, T_len)
    R_mean = []
    for i in range(n):
        R_i = []
        for j in range(T_len-W_len):
            idx = start+T_len*i+j
            if valid[idx]:
                R_i.append(R[idx])
        R_mean.append(np.mean(R_i))
    R_mean = np.array(R_mean)   
    return R_mean, a, b

def meas_AWG(ch_DAQ, wf_dict, chs_OSC, R_s=1e3, R_min=5e3, plot=True, save=False):
    ch_AWG = wf_dict['ch']
    
    DAQ = instruments['DAQ']
    AWG = instruments['AWG']
    OSC = instruments['OSC']
    
    T = wf_dict['T']
    W = wf_dict['W']
    n = wf_dict['n']
    read_V = wf_dict['read_V']
    read = wf_dict['read']
    SRAT = AWG.get_srat(ch_AWG)
    t_scale = T*sum(n)/4
    t_pos = T*sum(n)
    scales = [read_V/5, read_V*R_s/(R_s+R_min)/5]
    trig_source = chs_OSC[0]
    trig_level = read_V*.9
    OSC.configure(SRAT, t_scale, t_pos, chs_OSC, scales, trig_source, trig_level)
    DAQ.set_conn(ch_DAQ)
    AWG.set_outp(ch_AWG, 1)
    AWG.trigger()
    AWG.set_outp(ch_AWG, 0)
    
    wf = []
    if 'wf' not in os.listdir():
        os.mkdir('wf')
    for ch_OSC in chs_OSC:
        path = f"wf/{ch_OSC}.csv"
        with open(path, "w") as f:
            f.write(OSC.get_wf(ch_OSC))
        wf.append(np.transpose(np.loadtxt(path, delimiter=";")))
    
    T_len = round(T*SRAT)
    W_len = np.round(np.array(W)*SRAT).astype(int).min()
    
    if read and read_V > 0:
        R = R_s*(wf[0][1]-wf[1][1])/wf[1][1]
        valid = get_valid(wf[0][1], T_len, W_len, read_V, wf_o=wf[1][1])
        t = wf[0][0]
        V_i = wf[0][1]
        R_mean, a, b = get_R_mean(t, V_i, read_V, R, valid, T_len, W_len, sum(n)+1)
        #R_mean, a, b = get_R_mean_(R, valid, W_len)
        t_mean = (t[a]+t[b])/2
    
    results = {
        'wf' : [wf[i].tolist() for i in range(2)],
        'R_valid' : R[valid].tolist(),
        'valid' : valid.tolist(),
        'R_mean' : R_mean.tolist(),
        't_mean' : t_mean.tolist()
    } if read else {
        'wf' : [wf[i].tolist() for i in range(2)]
    }
    
    if plot:
        plot_results(results, 'AWG')
     
    if save:
        meas_dict = {
            **wf_dict,
            **results
        }
        save_meas(meas_dict, 'AWG')
        
    return results

def meas_SMU(ch_DAQ, wf_dict, chs_OSC, plot=True, save=False):
    T = wf_dict['T']
    W = wf_dict['W']
    n = wf_dict['n']
    read_V = wf_dict['read_V']
    SRAT = instruments['SMU'].get_srat(wf_dict['ch'])
    t_scale = T*sum(n)/4
    t_pos = T*sum(n)
    scales = [read_V/5, read_V/5]
    trig_source = chs_OSC[0]
    trig_level = read_V*.9
    instruments['OSC'].configure(SRAT, t_scale, t_pos, chs_OSC, scales, trig_source, trig_level)
    return meas_SMU_(ch_DAQ, wf_dict, plot=plot, save=save)

def meas_SMU_(ch_DAQ, wf_dict, plot=True, save=False):
    ch_SMU = wf_dict['ch']
    DAQ = instruments['DAQ']
    SMU = instruments['SMU']
    
    DAQ.set_conn(ch_DAQ)
    SMU.set_outp(ch_SMU, 1)
    SMU.trigger(ch_SMU)
    SMU.set_outp(ch_SMU, 0)
    
    V_i = np.array(SMU.query(f':fetc:arr:volt? (@{ch_SMU})').strip().split(',')).astype('float')
    I_o = np.array(SMU.query(f':fetc:arr:curr? (@{ch_SMU})').strip().split(',')).astype('float')
    t = np.array(SMU.query(f':fetc:arr:time? (@{ch_SMU})').strip().split(',')).astype('float')
    
    T = wf_dict['T']
    W = wf_dict['W']
    n = wf_dict['n']
    read_V = wf_dict['read_V']
    read = wf_dict['read']
    SRAT = SMU.get_srat(ch_SMU)
    T_len = round(T*SRAT)
    W_len = np.round(np.array(W)*SRAT).astype(int).max()
    
    aint = SMU.get_trig_source(ch_SMU) == 'AINT'
    if read:
        R = V_i / I_o
        if aint:
            valid = (V_i > .01)
        else:
            valid = get_valid(V_i, T_len, W_len, read_V)
            R_mean, a, b = get_R_mean(t, V_i, read_V, R, valid, T_len, W_len, sum(n)+1)
            #R_mean, a, b = get_R_mean(R, valid, W_len)
            t_mean = (t[a]+t[b])/2
    
    results = {
        't' : t.tolist(),
        'V_i' : V_i.tolist(),
        'I_o' : I_o.tolist(),
        'R_valid' : R[valid].tolist(),
        'valid' : valid.tolist(),
        'R_mean' : R_mean.tolist(),
        't_mean' : t_mean.tolist()
    } if read and not aint else {
        't' : t.tolist(),
        'V_i' : V_i.tolist(),
        'I_o' : I_o.tolist(),
        'R_valid' : R[valid].tolist(),
        'valid' : valid.tolist(),
    } if read > 0 else {
        't' : t.tolist(),
        'V_i' : V_i.tolist(),
        'I_o' : I_o.tolist()
    }
    if plot:
        plot_results(results, 'SMU', aint=aint)
    if save:
        meas_dict = {
            **wf_dict,
            **results
        }
        save_meas(meas_dict, 'SMU')
    
    return results

def meas_AWG_SMU(wf_dict_AWG_list, N, ch_DAQ_AWG, ch_DAQ_SMU, wf_dict_SMU, R_s = 1e3, plot=True, save=False):
    if not len(N) == len(wf_dict_AWG_list):
        raise RuntimeError("Inputs wf_dict_AWG_list and N of meas_AWG_SMU have different lengths.")
    
    get_and_set_wf('SMU', wf_dict_SMU, aint=True)
    
    p = 0
    P = []
    R = []
    P.append(p)
    results = meas_SMU_(ch_DAQ_SMU, wf_dict_SMU, plot=False)
    R.append(results['R_valid'][0] - R_s)
    for i, wf_dict_AWG in enumerate(wf_dict_AWG_list):
        get_and_set_wf('AWG', wf_dict_AWG)
        ch_AWG = wf_dict_AWG['ch']

        DAQ = instruments['DAQ']
        AWG = instruments['AWG']
        SMU = instruments['SMU']

        for n in range(N[i]):
            DAQ.set_conn(ch_DAQ_AWG)
            AWG.set_outp(ch_AWG, 1)
            AWG.trigger()
            AWG.set_outp(ch_AWG, 0)
            p -= wf_dict_AWG['n'][0]*int(np.sign(wf_dict_AWG['V'][0]))
            P.append(p)
            results = meas_SMU_(ch_DAQ_SMU, wf_dict_SMU, plot=False)
            R.append(results['R_valid'][0] - R_s)
    if plot:
        b = np.floor(len(P)/2).astype(int)
        plt.plot(P[:b+1], R[:b+1], '.-b')
        plt.plot(P[b:], R[b:], '.-r')
        plt.xlabel("signed pulse count")
        plt.ylabel("$R$ / $\Omega$")
        plt.show()
    results = {
        'P' : P,
        'R' : R
    }
    meas_dict = {
        'V' : [wf_dict_AWG['V'][0] for wf_dict_AWG in wf_dict_AWG_list],
        'n' : [wf_dict_AWG['n'][0] for wf_dict_AWG in wf_dict_AWG_list],
        'N' : N,
        **wf_dict_SMU,
        **results
    }
    if save:
        save_meas(meas_dict, 'AWG_SMU')
    return results

def plot_results(results, key, aint=False):
    if key == 'AWG':
        wf = results['wf']
        fig, ax0 = plt.subplots()
        ax1 = ax0.twinx()
        ax0.set_xlabel("$t$ / s")
        ax0.plot(wf[0][0], wf[0][1], "y")
        ax0.set_ylabel("$V$ / V")
        ax0.legend(["ch1"], loc="upper left")
        ax1.plot(wf[1][0], wf[1][1], "g")
        ax1.set_ylabel("$V$ / V")
        ax1.legend(["ch2"], loc="upper right")
        plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
        plt.show()
        
        t = np.array(wf[0][0])
        V_i = wf[0][1]
    elif key == 'SMU':
        t = np.array(results['t'])
        V_i = results['V_i']
    if "valid" in results.keys():
        valid = np.array(results['valid'])
        fig, ax0 = plt.subplots()
        ax1 = ax0.twinx()
        ax0.step(t, V_i, 'y', where='post')
        ax0.legend(["V_i"], loc="upper left")
        ax0.set_xlim(t[0], t[-1])
        ax0.set_xlabel("$t$ / s")
        ax0.set_ylabel("$V$ / V")
        ax1.plot(t[valid], results['R_valid'], '.g')
        if not aint:
            ax1.plot(results['t_mean'], results['R_mean'], '.r')
        ax1.set_ylabel("$R$ / $\Omega$")
        ax0.legend(["$V$"], loc="upper left")
        ax1.legend(["$R$", "$R\_mean$"], loc="upper right")
        plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
        plt.show()
    
def save_meas(meas_dict, key):
    with open(f"meas_dict/{key}_{date.today()}.json", "w") as json_file:
            json_file.write(json.dumps(meas_dict))

def get_params(P, R, plot=True):
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

def pulse_s(G, alfa, beta):
    G_inf = 1/(1-np.exp(-beta*N**alfa))
    return (-np.log(1-G/G_inf)/beta)**(1/alfa)

def pulse_r(G, alfa, beta):
    G_inf = 1-1/(1-np.exp(-beta*N**alfa))
    return N -(-np.log(1-(1-G)/(1-G_inf))/beta)**(1/alfa)

def pulse_n(G_c, G_n, popt):
    G_i = np.clip(G_c, 0, 1)
    G_f = np.clip(G_n, 0, 1)
    if G_f > G_i:
        return pulse_s(G_f, *popt['s']) - pulse_s(G_i, *popt['s'])
    else:
        return pulse_r(G_f, *popt['r']) - pulse_r(G_i, *popt['r'])

def characterize(n, V_lim):
    V = [*np.linspace(V_lim[0], V_lim[1], N), *np.linspace(V_lim[2], V_lim[3], N)]
    wf_dict_AWG = {
        'V' : V, # pulse voltages
        'n' : [1 for i in range(N*2)],   # pulse repetitions
        'W' : [2e-4 for i in range(N*2)],     # pulse widths
        'T' : 2e-3,     # pulse period
        'read' : True,
        'read_V' : read_V,
        'ch' : ch_AWG
    }
    get_and_set_wf('AWG', wf_dict_AWG)
    P = np.array([])
    P_i  = np.concatenate([np.arange(N), np.arange(-1, -N-1, -1)])
    R = []
    for i in range(n):
        P = np.concatenate([P, P_i])
        results_AWG = meas_AWG(113, wf_dict_AWG, [1, 2], R_s=1e3, R_min=5e3, plot = False)
        R.extend(results_AWG['R_mean'])
    return P, np.array(R)
    
def read(params, ch_DAQ_AWG, ch_DAQ_SMU):
    wf_dict_SMU = {
        'V' : [0],
        'n' : [0],
        'W' : [0],
        'T' : 1,
        'read' : True,
        'read_V' : read_V,
        'ch' : ch_SMU
    }
    results = meas_AWG_SMU([], [], ch_DAQ_AWG, ch_DAQ_SMU, wf_dict_SMU, R_s = 1e3, plot=False)
    return (results['R'][0]**-1/params['A'] - 1)/params['B'] * ss - (ss-1)/2

def write(params, ch_DAQ_AWG, ch_DAQ_SMU, G, V_lim):
    G_i = np.clip(read(params, ch_DAQ_AWG, ch_DAQ_SMU), 0, 1)
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
        get_and_set_wf('AWG', wf_dict_AWG)
        DAQ = instruments['DAQ']
        AWG = instruments['AWG']
        DAQ.set_conn(ch_DAQ_AWG)
        AWG.set_outp(ch_AWG, 1)
        AWG.trigger()
        AWG.set_outp(ch_AWG, 0)
    