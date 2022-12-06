import matplotlib.pyplot as plt
from instruments import *

def my_wf(V, n, T, DCYC, const_read=False, read_p=.1):
    if not len(V) == len(n) == len(T):
        raise RuntimeError("Inputs of get_Vt have different lengths.")
    for i in range(len(V)):
        if i == 0:
            n_read = round(T[i]*(1-DCYC[i]))
            n_zeros = T[i]//2
            wf = [0 for j in range(n_zeros)] + [(read_p if const_read else 0) for j in range(n_read)]
        for j in range(n[i]):
            wf += [V[i] for k in range(round(T[i]*DCYC[i]))]
            wf += [(read_p if const_read else 0) for k in range(round(T[i]*(1-DCYC[i])))]
    wf += [0 for j in range(n_zeros)]
    return wf


def get_valid(wf, w, read_V, tol=.1):
    temp = np.convolve(wf, np.ones(w)/w, "same")
    return np.logical_and(read_V * (1-tol) < temp, temp < read_V * (1+tol))


def meas_AWG(DAQ, ch_DAQ, AWG, ch_AWG, T_len, read_V, VOLT, OSC, chs_OSC, plot=True):
    SRAT_AWG = AWG.get_srat(ch_AWG)
    t_scale = T_len/SRAT_AWG*2
    t_pos = T_len/SRAT_AWG*8
    scales = [VOLT/5, read_V/5]
    trig_source = chs_OSC[0]
    trig_level = read_V*.9
    OSC.configure(SRAT_AWG, t_scale, t_pos, chs_OSC, scales, trig_source, trig_level)
    
    
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
        
    
    R_p = 10e3
    R = R_p*(wf[0][1]-wf[1][1])/wf[1][1]
    valid = get_valid(wf[0][1], T_len//5, read_V)
    
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
        ax1.semilogy(wf[1][0, valid], R[valid], "g.")
        ax1.set_ylim([1, 1e9])
        ax1.set_ylabel("$R$ / $\Omega$")
        ax1.legend(["R"], loc="upper right")
        plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
        plt.show()
    
    return wf, R, valid

def meas_SMU(DAQ, ch_DAQ, SMU, ch_SMU, T_len, read_V, VOLT, OSC, chs_OSC, plot=True, aint=False):
    SRAT_SMU = SMU.get_srat(ch_SMU)
    t_scale = T_len/SRAT_SMU*2
    t_pos = T_len/SRAT_SMU*8
    if aint:
        t_scale *= 10
        t_pos = 0
    scales = [VOLT/5, read_V/5]
    trig_source = chs_OSC[0]
    trig_level = read_V*.9
    OSC.configure(SRAT_SMU*1e2, t_scale, t_pos, chs_OSC, scales, trig_source, trig_level)
    
    return meas_SMU_(DAQ, ch_DAQ, SMU, ch_SMU, T_len, read_V, plot=plot, aint=aint)


def meas_SMU_(DAQ, ch_DAQ, SMU, ch_SMU, T_len, read_V, plot=True, aint=False):
    DAQ.set_conn(ch_DAQ)
    SMU.set_outp(ch_SMU, 1)
    SMU.trigger()
    SMU.set_outp(ch_SMU, 0)
    
    V_i = np.array(SMU.query(f':fetc:arr:volt? (@{ch_SMU})').strip().split(',')).astype('float')
    I_o = np.array(SMU.query(f':fetc:arr:curr? (@{ch_SMU})').strip().split(',')).astype('float')
    t = np.array(SMU.query(f':fetc:arr:time? (@{ch_SMU})').strip().split(',')).astype('float')
    
    R = V_i / I_o
    valid = (V_i > .01) if aint else get_valid(V_i, T_len//5, read_V)
    
    if plot:
        fig, ax0 = plt.subplots()
        ax1 = ax0.twinx()
        ax0.step(t, V_i, where='post')
        ax0.set_xlim(t[0], t[-1])
        ax0.set_xlabel("$t$ / s")
        ax0.set_ylabel("$V$ / V")
        ax1.semilogy(t[valid], R[valid], 'g.')
        ax1.set_ylim([1, 1e9])
        ax1.set_ylabel("$R$ / $\Omega$")
        ax0.legend(["$V$"], loc="upper left")
        ax1.legend(["$R$"], loc="upper right")
        plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
        plt.show()

    return t, V_i, I_o, R, valid

def meas_AWG_SMU(N, DAQ, AWG, ch_AWG, SMU, ch_SMU, T_len, read_V):
    R_array = []
    R_p = 9750
    for n in range(N):
        DAQ.set_conn(113)
        AWG.set_outp(ch_AWG, 1)
        AWG.trigger()
        AWG.set_outp(ch_AWG, 0)
        t, V_i, I_o, R, valid = meas_SMU_(DAQ, 111, SMU, ch_SMU, T_len, read_V, plot=False, aint=True)    
        R_array.append(R[valid] - R_p)
    R = [np.mean(R_array[n]) for n in range(N)]
    return R
