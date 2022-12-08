import matplotlib.pyplot as plt
import json
from instruments import *

instruments = {}
rm = pyvisa.ResourceManager()
lr = rm.list_resources()
for r in lr:
    if "awg" in r:
        instruments['AWG'] = ArbitraryWaveformGenerator(r)
    elif "daq" in r:
        instruments['DAQ'] = DataAcquisition(r)
    elif "172.31.182.32" in r:
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

def my_wf(V, n, T, W, read=False, read_V=.1):
    SRAT = 10/W if W else 1
    T_len = round(T*SRAT)
    W_len = round(W*SRAT)
    if not len(V) == len(n):
        raise RuntimeError("Inputs of my_wf have different lengths.")
    for i in range(len(V)):
        if i == 0:
            n_read = T_len-W_len
            n_zeros = T_len//2
            wf = [0 for j in range(n_zeros)] + [(read_V if read else 0) for j in range(n_read)]
        for j in range(n[i]):
            wf += [V[i] for k in range(W_len)]
            wf += [(read_V if read else 0) for k in range(T_len - W_len)]
    wf += [0 for j in range(n_zeros)]
    return wf, SRAT

def get_and_set_wf(key, ch, wf_dict, aint=False):
    inst = instruments[key]
    wf, SRAT = my_wf(wf_dict['V'], wf_dict['n'], wf_dict['T'], wf_dict['W'], wf_dict['read'], wf_dict['read_V'])
    if aint:
        inst.set_wf(wf, SRAT, ch, aint=aint)
    else:
        inst.set_wf(wf, SRAT, ch)

def get_valid(wf_i, T_len, W_len, read_V, tol=.05, wf_o=[]):
    a = np.zeros(round(W_len*0.8))
    b = np.ones(T_len//5)/(T_len//5)
    w = np.concatenate([a, b])
    temp = np.convolve(wf_i, w, "same")
    valid = np.logical_and(read_V * (1-tol) < temp, temp < read_V * (1+tol))
    if len(wf_o):
        temp = abs(np.gradient(np.convolve(wf_o, w, "same"))) < 1e-4*read_V
        valid = np.logical_and(valid, temp)
    return valid

def get_R_mean(R, valid, W_len):
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


def meas_AWG(ch_DAQ, ch_AWG, wf_dict, chs_OSC, R_s=9750, plot=True, save=True):
    DAQ = instruments['DAQ']
    AWG = instruments['AWG']
    OSC = instruments['OSC']
    
    T = wf_dict['T']
    W = wf_dict['W']
    read_V = wf_dict['read_V']
    SRAT = AWG.get_srat(ch_AWG)
    t_scale = T*2
    t_pos = T*8
    scales = [read_V/5, read_V/5]
    trig_source = chs_OSC[0]
    trig_level = read_V*.9
    OSC.configure(SRAT, t_scale, t_pos, chs_OSC, scales, trig_source, trig_level)
    
    DAQ.set_conn(ch_DAQ)
    AWG.set_outp(ch_AWG, 1)
    AWG.trigger()
    AWG.set_outp(ch_AWG, 0)
    
    wf = []
    for ch_OSC in chs_OSC:
        f = open(f"wf{ch_OSC}.csv", "w")
        f.write(OSC.get_wf(ch_OSC))
        f.close()
        wf.append(np.transpose(np.loadtxt(f"wf{ch_OSC}.csv", delimiter=";")))
    
    T_len = round(T*SRAT)
    W_len = round(W*SRAT)
    
    R = R_s*(wf[0][1]-wf[1][1])/wf[1][1]
    valid = get_valid(wf[0][1], T_len, W_len, read_V, wf_o=wf[1][1])
    t = wf[1][0]
    R_mean, a, b = get_R_mean(R, valid, W_len)
    t_mean = (t[a]+t[b])/2
    
    if plot:
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

        fig, ax0 = plt.subplots()
        ax1 = ax0.twinx()
        ax0.set_xlabel("$t$ / s")
        ax0.plot(wf[0][0], wf[0][1], "y")
        ax0.set_ylabel("$V$ / V")
        ax0.legend(["ch1"], loc="upper left")
        ax1.semilogy(t[valid], R[valid], ".g")
        ax1.semilogy(t_mean, R_mean, '.r')
        ax1.set_ylim([1, 1e9])
        ax1.set_ylabel("$R$ / $\Omega$")
        ax1.legend(["$R$", "$R\_mean$"], loc="upper right")
        plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
        plt.show()
        
    results = {
        'wf' : [wf[i].tolist() for i in range(2)],
        'R_valid' : R[valid].tolist(),
        'valid' : valid.tolist(),
        'R_mean' : R_mean.tolist(),
        't_mean' : t_mean.tolist()
    }
    if save:
        meas_dict = {
            **wf_dict,
            **results
        }
        with open("meas_dict_AWG.json", "w") as json_file:
            json_file.write(json.dumps(meas_dict))
    return results

def meas_SMU(ch_DAQ, ch_SMU, wf_dict, chs_OSC, plot=True, save=True):
    T = wf_dict['T']
    W = wf_dict['W']
    read_V = wf_dict['read_V']
    SRAT = instruments['SMU'].get_srat(ch_SMU)
    t_scale = T*2
    t_pos = T*8
    scales = [read_V/5, read_V/5]
    trig_source = chs_OSC[0]
    trig_level = read_V*.9
    instruments['OSC'].configure(SRAT*1e2, t_scale, t_pos, chs_OSC, scales, trig_source, trig_level)
    return meas_SMU_(ch_DAQ, ch_SMU, wf_dict, plot=plot, save=save)


def meas_SMU_(ch_DAQ, ch_SMU, wf_dict, plot=True, save=True):
    DAQ = instruments['DAQ']
    SMU = instruments['SMU']
    
    DAQ.set_conn(ch_DAQ)
    SMU.set_outp(ch_SMU, 1)
    SMU.trigger()
    SMU.set_outp(ch_SMU, 0)
    
    V_i = np.array(SMU.query(f':fetc:arr:volt? (@{ch_SMU})').strip().split(',')).astype('float')
    I_o = np.array(SMU.query(f':fetc:arr:curr? (@{ch_SMU})').strip().split(',')).astype('float')
    t = np.array(SMU.query(f':fetc:arr:time? (@{ch_SMU})').strip().split(',')).astype('float')
    
    T = wf_dict['T']
    W = wf_dict['W']
    read_V = wf_dict['read_V']
    SRAT = SMU.get_srat(ch_SMU)
    T_len = round(T*SRAT)
    W_len = round(W*SRAT)
    
    R = V_i / I_o
    aint = SMU.get_trig_source(ch_SMU) == 'AINT'
    if aint:
        valid = (V_i > .01)
    else:
        valid = get_valid(V_i, T_len, W_len, read_V)
        R_mean, a, b = get_R_mean(R, valid, W_len)
        t_mean = (t[a]+t[b])/2
    
    
    if plot:
        fig, ax0 = plt.subplots()
        ax1 = ax0.twinx()
        ax0.step(t, V_i, where='post')
        ax0.set_xlim(t[0], t[-1])
        ax0.set_xlabel("$t$ / s")
        ax0.set_ylabel("$V$ / V")
        ax1.semilogy(t[valid], R[valid], '.g')
        if not aint:
            ax1.semilogy(t_mean, R_mean, '.r')
        ax1.set_ylim([1, 1e9])
        ax1.set_ylabel("$R$ / $\Omega$")
        ax0.legend(["$V$"], loc="upper left")
        ax1.legend(["$R$", "$R\_mean$"], loc="upper right")
        plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
        plt.show()
        
    results = {
        't' : t.tolist(),
        'V_i' : V_i.tolist(),
        'I_o' : I_o.tolist(),
        'R_valid' : R[valid].tolist(),
        'valid' : valid.tolist(),
        'R_mean' : R_mean.tolist(),
        't_mean' : t_mean.tolist()
    } if not aint else {
        't' : t.tolist(),
        'V_i' : V_i.tolist(),
        'I_o' : I_o.tolist(),
        'R_valid' : R[valid].tolist(),
        'valid' : valid.tolist()
    }
    if save:
        meas_dict = {
            **wf_dict,
            **results
        }
        with open("meas_dict_SMU.json", "w") as json_file:
            json_file.write(json.dumps(meas_dict))
    return results

def meas_AWG_SMU(N, ch_AWG, ch_SMU, wf_dict_SMU, R_s = 9750, plot=True):
    DAQ = instruments['DAQ']
    AWG = instruments['AWG']
    SMU = instruments['SMU']
    
    R_array = []
    for n in range(N):
        DAQ.set_conn(113)
        AWG.set_outp(ch_AWG, 1)
        AWG.trigger()
        AWG.set_outp(ch_AWG, 0)
        results = meas_SMU_(111, ch_SMU, wf_dict_SMU, plot=False)    
        R_array.append(np.array(results['R_valid']) - R_s)
    R = [np.mean(R_array[n]) for n in range(N)]
    if plot:
        plt.plot(R, '.')
        plt.ylabel("$R$ / $\Omega$")
        plt.show()
    return R
