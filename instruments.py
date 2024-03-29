import pyvisa
import numpy as np


def connect(instruments):
    for key in instruments.keys():
        instruments[key].connect()

def disconnect(instruments):
    for key in instruments.keys():
        instruments[key].disconnect()
        
def reset(instruments):
    for key in instruments.keys():
        if instruments[key].connected:
            instruments[key].reset()

        
class Instrument:
    def __init__(self, address):
        self.address = address
        self.connected = False

    def connect(self):
        if not self.connected:
            try:
                self.conn = pyvisa.ResourceManager().open_resource(self.address)
                self.connected = True
            except:
                pass

    def disconnect(self):
        if self.connected:
            self.conn.close()
            self.connected = False

    def reset(self):
        self.write("*RST")

    def get_id(self):
        return self.query("*IDN?")

    def query(self, command):
        if self.connected:
            self.conn.write(command)
            while True:
                try:
                    return self.conn.read().strip()
                except:
                    pass

    def write(self, command):
        if self.connected:
            self.conn.write(command)
            self.conn.write("*OPC?")
            while True:
                try:
                    self.conn.read()
                    break
                except:
                    pass
            
    
    def write_multiple(self, commands):
        for command in commands:
            self.write(command)
            

class SourceMeasureUnit(Instrument):
    def __init__(self, addr):
        super().__init__(addr)
    
    def get_srat(self, ch):
        return 1/float(self.query(f"sour{ch}:puls:widt?"))
    
    def get_trig_source(self, ch):
        return self.query(f"TRIG{ch}:TRAN:SOUR?")
    
    def set_trig_source(self, ch, source):
        self.write(f"TRIG{ch}:SOUR {source}")
    
    def set_wf(self, wave, SRAT, ch, prot_pos=1e-3, prot_neg=1e-3, aint=False):
        wf = ''
        for value in wave:
            wf += f'{value}, '
        wf += '0'
        rang = max(prot_pos, prot_neg)
        
        if aint:
            self.write_multiple(
                [
                    f'sour{ch}:func:mode volt',
                    f'sour{ch}:volt:mode list',
                    f'sour{ch}:list:volt ' + wf,
                    f'sens{ch}:func ""curr""',
                    f'sens{ch}:curr:rang:auto on',
                    f'sens{ch}:curr:rang:auto:llim 1e-7',
                    f'sens{ch}:curr:nplc 0.1',
                    f'sens{ch}:curr:prot ' + str(rang),
                    f'trig{ch}:coun ' + str(len(wf.split(',')))
                ]
            )
            self.set_trig_source(ch, "aint")
        else:
            tim = 1/SRAT
            self.write_multiple(
                [
                    f'sour{ch}:func:mode volt',
                    f'sour{ch}:volt:mode list',
                    f'sour{ch}:list:volt ' + wf,
                    f'sour{ch}:puls:widt ' + str(tim),
                    f'sens{ch}:func ""curr""',
                    f'sens{ch}:curr:rang:auto off',
                    f'sens{ch}:curr:rang ' + str(rang),
                    f'sens{ch}:curr:aper ' + str(tim/2),
                    f'trig{ch}:acq:del ' + str(tim/2),
                    f'trig{ch}:tim ' + str(tim),
                    f'trig{ch}:coun ' + str(len(wf.split(',')))
                ]
            )
            self.set_curr_prot(ch, 'pos', prot_pos)
            self.set_curr_prot(ch, 'neg', prot_neg)
            self.set_trig_source(ch, "tim")
    
    def get_curr_prot(self, ch, pol):
        return float(self.query(f'sens{ch}:curr:prot:{pol}?'))
    
    def set_curr_prot(self, ch, pol, prot):
        self.write(f'sens{ch}:curr:prot:{pol} {prot}')
    
    def get_outp(self, ch):
        return int(self.query(f':outp{ch}?'))
    
    def set_outp(self, ch, out):
        self.write(f':outp{ch} {out}')
    
    def trigger(self, ch):
        self.write(f':init (@{ch})')
    

class ArbitraryWaveformGenerator(Instrument):
    def __init__(self, addr):
        super().__init__(addr)
    
    def get_srat(self, ch):
        return float(self.query(f'SOUR{ch}:FUNC:ARB:SRAT?'))
    
    def set_wf(self, wave, SRAT, ch):
        VOLT = max(wave)-min(wave)
        wf = np.concatenate([np.array(wave)/VOLT, [0]])
        wf_bytes = wf.astype('single').tobytes()
        wf_len = str(len(wf_bytes))
        self.write_multiple(
            [
                'FORM:BORD SWAP',
                f'SOUR{ch}:DATA:VOL:CLE'
            ]
        )
        self.conn.write_raw((f'SOUR{ch}:DATA:ARB myArb, #'+str(len(wf_len))+wf_len).encode()+wf_bytes)
        self.write_multiple(
            [
                f'SOUR{ch}:FUNC ARB',
                f'SOUR{ch}:FUNC:ARB:SRAT {SRAT}',
                f'SOUR{ch}:FUNC:ARB myArb',
                f'OUTP{ch}:LOAD INF',
                f'SOUR{ch}:VOLT {VOLT}',
                #f'SOUR{ch}:VOLT:OFFS 0',
                f'TRIG{ch}:SOUR BUS',
                f'SOUR{ch}:BURS:MODE TRIG',
                f'SOUR{ch}:BURS:STAT ON',
                #f'SOUR{ch}:BURS:NCYC 1',
            ]
        )
    
    def get_outp(self, ch):
        return int(self.query(f':outp{ch}?'))
    
    def set_outp(self, ch, out):
        self.write(f':outp{ch} {out}')
    
    def trigger(self):
        self.write('*TRG')


class DataAcquisition(Instrument):
    def __init__(self, addr):
        super().__init__(addr)
        self.closed = [0, 0]
        
    def set_conn(self, ch):
        b = (ch // 10) % 10 - 1
        c = ch % 10
        if self.closed[b] != c:
            self.write(f"ROUT:CLOS:EXCL (@{ch})")
            self.closed[b] = c
        
class Oscilloscope(Instrument):
    def __init__(self, addr):
        super().__init__(addr)
        
    def get_hdef_state(self):
        return self.query("HDEFinition:STATe?")
    
    def set_hdef_state(self, state):
        self.write(f"HDEFinition:STATe {state}")
    
    def get_hdef_bwidth(self):
        return self.query("HDEFinition:BWIDth?")
    
    def set_hdef_bwidth(self, bwidth):
        self.write(f"HDEFinition:BWIDth {bwidth}")
        
    def get_ch(self, ch):
        return self.query(f"CHANnel{ch}:WAVeform1:STATe?")
    
    def set_ch(self, ch, state):
        self.write(f"CHANnel{ch}:WAVeform1:STATe {state}")
        
    def get_ch_scale(self, ch):
        return float(self.query(f"CHANnel{ch}:SCALe?"))
    
    def set_ch_scale(self, ch, scale):
        self.write(f"CHANnel{ch}:SCALe {scale}")
        
    def get_ch_pos(self, ch):
        return float(self.query(f"CHANnel{ch}:POSition?"))
    
    def set_ch_pos(self, ch, pos):
        return self.write(f"CHANnel{ch}:POSition {pos}")
        
    def get_res(self):
        return float(self.query("ACQuire:RESolution?"))
    
    def set_res(self, res):
        self.write(f"ACQuire:RESolution {res}")
    
    def get_srat(self):
        return float(self.query("ACQuire:SRATe?"))
    
    def set_srat(self, srat):
        self.write(f"ACQuire:SRATe {srat}")
    
    def get_points(self):
        return int(self.query("ACQuire:POINts?"))
    
    def set_points(self, points):
        self.write(f"ACQuire:POINts {points}")
    
    def get_time_scale(self):
        return float(self.query("TIMebase:SCALe?"))
        
    def set_time_scale(self, scale):
        self.write(f"TIMebase:SCALe {scale}")
        
    def get_time_pos(self):
        self.write("TIMebase:HORizontal:POSition?")
    
    def set_time_pos(self, pos):
        self.write(f"TIMebase:HORizontal:POSition {pos}")
    
    def get_trig_source(self):
        return self.query(f"TRIGger:SOURce?")
    
    def set_trig_source(self, ch):
        self.write(f"TRIGger:SOURce CHAN{ch}")
    
    def get_trig_mode(self):
        return self.query(f"TRIGger:MODE?")
        
    def set_trig_mode(self, mode):
        self.write(f"TRIGger:MODE {mode}")
    
    def get_trig_edge_slope(self):
        return self.query(f"TRIGger:EDGE:SLOPe?")
        
    def set_trig_edge_slope(self, slope):
        self.write(f"TRIGger:EDGE:SLOPe {slope}")
    
    def get_trig_level(self):
        return self.query(f"TRIGger:LEVel?")
        
    def set_trig_level(self, level):
        self.write(f"TRIGger:LEVel {level}")
    
    def configure(self, SRAT, t_scale, t_pos, chs, scales, trig_source, trig_level):
        self.set_hdef_state('ON')
        self.set_hdef_bwidth(1e5)
        self.set_time_scale(t_scale)
        self.set_time_pos(t_pos)
        self.set_srat(SRAT)
        for ch, scale in zip(chs, scales):
            self.set_ch(ch, 'ON')
            self.set_ch_scale(ch, scale)
            self.set_ch_pos(ch, -2.5)
        self.set_trig_mode('NORM')
        self.set_trig_source(trig_source)
        self.set_trig_level(trig_level)
    
    def get_wf(self, ch):
        self.write_multiple(
            [
                f"EXPort:WAVeform:SOURce C{ch}W1",
                "EXPort:WAVeform:SCOPe WFM",
                "EXPort:WAVeform:NAME 'C:\Data\DataExportWfm_analog.csv'",
                "EXPort:WAVeform:RAW OFF",
                "EXPort:WAVeform:INCXvalues ON",
                "EXPort:WAVeform:DLOGging OFF",
                "MMEM:DEL 'C:\Data\DataExportWfm_analog.*'",
                "EXPort:WAVeform:SAVE"
            ]
        )
        return self.query("MMEM:DATA? 'C:\Data\DataExportWfm_analog.wfm.csv'")
    
    

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

ch_AWG, ch_SMU = 1, 2
ch_DAQ_AWG, ch_DAQ_SMU = 113, 111
