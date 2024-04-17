# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 10:15:32 2018

@author: shussai
"""

cimport cython
cimport numpy as npc
import numpy as np

cdef extern from "math.h":\

    cdef double exp(double x)
    cdef double log(double x)
    cdef double pow(double x, double y)
    cdef double sqrt(double x) nogil    
    cdef double tanh(double x) nogil    
    
cdef class TenTuscher:
    
    cdef public double R 				
    cdef public double T 				
    cdef public double F 				
    cdef public double Cm 
    cdef public double Cc 			  
    cdef public double S_V_ratio 	           
    cdef public double rho 			
    cdef public double V_c 			  
    cdef public double V_SR 			  
    cdef public double pi 
    cdef public double dt 					
    cdef public double G_Na 
    cdef public double G_K1
    cdef public double G_to_epi_M
    cdef public double G_to_endo 
    cdef public double G_Kr 
    cdef public double G_Ks_epi_endo 
    cdef public double G_Ks_M
    cdef public double G_CaL 
    cdef public double G_pK 
    cdef public double G_pCa 
    cdef public double G_bNa 
    cdef public double G_bCa
    cdef public double p_KNa		  
    cdef public double P_NaK 		
    cdef public double k_NaCa	  
    cdef public double k_sat			      	
    cdef public double K_mNai  
    cdef public double K_mNa		
    cdef public double K_mK 	   
    cdef public double K_mCa	 
    cdef public double K_pCa     
    cdef public double K_up 	 
    cdef public double K_bufsr 
    cdef public double K_bufc   
    cdef public double V_leak 	
    cdef public double V_maxup 	 
    cdef public double Buf_c
    cdef public double Buf_sr 		
    cdef public double a_rel 	  
    cdef public double b_rel 			
    cdef public double c_rel 	    
    cdef public double alpha 		
    cdef public double gama 			
    cdef public double stim 
    cdef public double Na_o
    cdef public double K_o
    cdef public double Ca_o 
   


    cdef public double Na_i
    cdef public double Ca_i
    cdef public double K_i
    cdef public double Ca_sr
    cdef public double m
    cdef public double h
    cdef public double j
    cdef public double d
    cdef public double f
    cdef public double fCa
    cdef public double r
    cdef public double s
    cdef public double xs
    cdef public double xr1
    cdef public double xr2
    cdef public double g
    cdef public double V
    cdef public double t
    cdef public double dx
    cdef public double dy    

#ChR2
    cdef public double E_ChR2 
    cdef public double g_ChR2
    cdef public double G_d2 
    cdef public double e_12d 
    cdef public double c1_12d 
    cdef public double c2_12d 
    cdef public double e_21d 
    cdef public double c1_21d 
    cdef public double c2_21d
    cdef public double epsilon 
    cdef public double epsilon1
    cdef public double epsilon2
    cdef public double tau_ChR2 
    cdef public double sigma_ret 
    cdef public double landa 
    cdef public double w_loss
    cdef public double hc 
    cdef public double gamma

#    ChR2 dynamical variables
    cdef public double IChR_O1
    cdef public double IChR_O2
    cdef public double IChR_C1
    cdef public double IChR_C2
    cdef public double IChR_p
    cdef public double IChR2
    cdef public double I 

    @property
    def t_arr(self):
        return np.asarray(self.t)
    @property
    def V_arr(self):
        return np.asarray(self.V)
    @property
    def Na_i_arr(self):
        return np.asarray(self.Na_i)
    @property
    def Ca_i_arr(self):
        return np.asarray(self.Ca_i)
    @property
    def K_i_arr(self):
        return np.asarray(self.K_i)
    @property
    def Ca_sr_arr(self):
        return np.asarray(self.Ca_sr)
    @property
    def m_arr(self):
        return np.asarray(self.m)
    @property
    def h_arr(self):
        return np.asarray(self.h)
    @property
    def j_arr(self):
        return np.asarray(self.j)
    @property
    def d_arr(self):
        return np.asarray(self.d)
    @property
    def f_arr(self):
        return np.asarray(self.f)
    @property
    def fCa_arr(self):
        return np.asarray(self.fCa)
    @property
    def r_arr(self):
        return np.asarray(self.r)
    @property
    def s_arr(self):
        return np.asarray(self.s)
    @property
    def xs_arr(self):
        return np.asarray(self.xs)
    @property
    def xr1_arr(self):
        return np.asarray(self.xr1)
    @property
    def xr2_arr(self):
        return np.asarray(self.xr2)
    @property
    def g_arr(self):
        return np.asarray(self.g)
    @property
    def IChR2_arr(self):
        return np.asarray(self.IChR2) 



    
    def __init__ (self):
        
        self.R = 8314.472
        self.T = 310
        self.F = 96485.3415
        self.Cm = 1.
        self.Cc = 0.185
        self.S_V_ratio = 0.2
        self.rho = 162
        self.V_c = 0.016404
        self.V_SR = 0.001094
        self.pi = 3.141592654
        
        self.G_Na = 14.838
        self.G_K1 = 5.405
        self.G_to_epi_M = 0.0
        self.G_to_endo = 0.073
        self.G_Kr = 0.096
        self.G_Ks_epi_endo = 0.245
        self.G_Ks_M = 0.062
        self.G_CaL = 0.000175
        self.G_pK = 0.0146
        self.G_pCa = 0.825
        self.G_bNa = 0.00029
        self.G_bCa = 0.000592
        
      
        self.Na_o = 140.0
        self.K_o  = 5.4
        self.Ca_o = 2.0
            
        self.p_KNa     = 0.03			  
        self.P_NaK     = 1.362			  
        self.k_NaCa    = 1000.0		     
        self.k_sat     = 0.1			  
        self.K_mNai    = 87.5			 
        self.K_mNa     = 40.0			
        self.K_mK      = 1.0			 
        self.K_mCa     = 1.38			
        self.K_pCa     = 0.0005		
        self.K_up      = 0.00025	
        self.K_bufsr   = 0.3
        self.K_bufc    = 0.001		   
        self.V_leak    = 0.00008		   
        self.V_maxup   = 0.000425			   
        self.Buf_c     = 0.15			
        self.Buf_sr    = 10.0			
        self.a_rel     = 0.016464	    
        self.b_rel     = 0.25			
        self.c_rel     = 0.008232	   
        self.alpha     = 2.5			
        self.gama      = 0.35			
        self.stim      =  0
#        self.dur       = 2
#        self.incr_freq = 0.01		
#        self.step2     = 100
#        self.clock_threshold = 0.75
        self.dt              = 0.02
        
        
        self.Na_i  =  11.6
        self.Ca_i  =  0.0002
        self.K_i   =  138.3
        self.Ca_sr =  0.2
        self.m  =  0.0
        self.h  =  0.75
        self.j  =  0.75
        self.d  =  0.0
        self.f  =  1.0
        self.fCa=  1.0
        self.r  =  0.0
        self.s  =  1.0
        self.xs =  0.0
        self.xr1=  0.0
        self.xr2=  1.0
        self.g  =  1.0
        self.V  =  -86.2

#ChR2
        self.E_ChR2 = 0.
        self.g_ChR2 = 0.4
        self.G_d2 = 0.05
        self.e_12d = 0.011
        self.c1_12d = 0.005
        self.c2_12d = 0.024
        self.e_21d = 0.008
        self.c1_21d = 0.004
        self.c2_21d = 0.024
        self.epsilon = 0.5
        self.epsilon1 = 0.8535
        self.epsilon2 = 0.14
        self.tau_ChR2 = 1.3
        self.sigma_ret = 12e-20
        self.landa = 470
        self.w_loss = 1.3#0.77
        self.hc = 1.986446e-25
        self.gamma=.1

#       ChR2 vars               
        self.IChR_O1 = 0.
        self.IChR_O2 = 0.
        self.IChR_C1 = 1
        self.IChR_C2 = 0.
        self.IChR_p  = 0
        self.IChR2   = 0
        self.I       = 0

               
    cpdef local_step(self):
        
        cdef double R  = self.R			
        cdef double T  = self.T					
        cdef double F  = self.F	
        cdef double Cm = self.Cm
        cdef double Cc = self.Cc					   
        cdef double S_V_ratio =self.S_V_ratio       
        cdef double rho  = self.rho				 
        cdef double V_c  = self.V_c				
        cdef double V_SR = self.V_SR 	   
        cdef double pi   = self.pi
        cdef double dt   = self.dt	
        cdef double G_Na = self.G_Na
        cdef double G_K1 = self.G_K1
        cdef double G_to_epi_M = self.G_to_epi_M
        cdef double G_to_endo  = self.G_to_endo
        cdef double G_Kr = self.G_Kr
        cdef double G_Ks_epi_endo = self.G_Ks_epi_endo
        cdef double G_Ks_M = self.G_Ks_M
        cdef double G_CaL  = self.G_CaL
        cdef double G_pK   =self.G_pK
        cdef double G_pCa  = self.G_pCa
        cdef double G_bNa  = self.G_bNa
        cdef double G_bCa  = self.G_bCa
        cdef double p_KNa  = self.p_KNa	  
        cdef double P_NaK  = self.P_NaK		    
        cdef double k_NaCa = self.k_NaCa	  
        cdef double k_sat  = self.k_sat		   
        cdef double K_mNai = self.K_mNai		
        cdef double K_mNa  = self.K_mNa		
        cdef double K_mK   = self.K_mK			     
        cdef double K_mCa  = self.K_mCa	
        cdef double K_pCa  = self.K_pCa    
        cdef double K_up   = self.K_up	 
        cdef double K_bufsr = self.K_bufsr
        cdef double K_bufc  = self.K_bufc	   
        cdef double V_leak  = self.V_leak
        cdef double V_maxup = self.V_maxup	
        cdef double Buf_c   = self.	Buf_c	  
        cdef double Buf_sr  = self.Buf_sr	
        cdef double a_rel   = self.a_rel      
        cdef double b_rel   = self.b_rel 
        cdef double c_rel   = self.c_rel     
        cdef double alpha   = self.alpha 		
        cdef double gama    =self.gama			 
        cdef double stim    = self.stim
#        cdef double dur     =self.dur
#        cdef double incr_freq = self.incr_freq	 
#        cdef double step2 = self.step2
#        cdef double clock_threshold = self.clock_threshold
        cdef double t         =   self.t
        cdef double dx        =   self.dx
        cdef double dy        =   self.dy
#        cdef double D         =   self.D
        cdef double Na_o      = self.Na_o
        cdef double K_o       = self.K_o
        cdef double Ca_o      = self.Ca_o
        
        cdef double Na_i
        cdef double Ca_i
        cdef double K_i
        cdef double Ca_sr
        cdef double m
        cdef double h
        cdef double j
        cdef double d
        cdef double f
        cdef double fCa
        cdef double r
        cdef double s
        cdef double xs
        cdef double xr1
        cdef double xr2
        cdef double g
        cdef double V

#        ChR2
        cdef double IChR_O1
        cdef double IChR_O2
        cdef double IChR_C1
        cdef double IChR_C2
        cdef double IChR_p
        cdef double IChR2
        cdef double I
        
        
        
        Na_i  = self.Na_i
        Ca_i  = self.Ca_i
        K_i   = self.K_i
        Ca_sr = self.Ca_sr
        m = self.m
        h = self.h
        j = self.j
        d = self.d
        f = self.f
        fCa = self.fCa
        r   = self.r
        s   = self.s
        xs  = self.xs
        xr1 = self.xr1
        xr2 = self.xr2
        g = self.g
        V = self.V
        
        

#       ChR2 vars
               
        IChR_O1 = self.IChR_O1 
        IChR_O2 = self.IChR_O2
        IChR_C1 = self.IChR_C1 
        IChR_C2 = self.IChR_C2 
        IChR_p = self.IChR_p  
        IChR2 = self.IChR2   
        I = self.I      
#ChR2
        
        tetha = 100*I
        F_ChR = (.0006*I*self.landa)/self.w_loss
#        F_ChR = (sigma_ret*I*landa)/(w_loss*hc)
        S0_tetha = 0.5*(1+tanh(120*(tetha-0.1)))
        e_12 = self.e_12d + self.c1_12d*log(1+I/self.c2_12d)
        e_21 = self.e_21d + self.c1_21d*log(1+I/self.c2_21d)        
        G_d1 = 0.075 + 0.043 * tanh((V+20.)/-20.)
        G_r = 4.34e-5 *exp(-.0211539274*V)
        k_1 = self.epsilon1*F_ChR*IChR_p       
        k_2 = self.epsilon2*F_ChR*IChR_p
        G_V = (10.6408 - 14.6408*exp(-V/42.7671))        
#        G_V = (1 - np.exp(-V/40.))*(1./15)
        IChR2 = self.g_ChR2*G_V*(IChR_O1+self.gamma*IChR_O2)        

        IChR_O1=(k_1*IChR_C1 - (G_d1 + e_12)*IChR_O1 + e_21*IChR_O2)*dt+IChR_O1
        IChR_O2=(k_2*IChR_C2 - (self.G_d2 + e_21)*IChR_O2 + e_12*IChR_O1)*dt+IChR_O2
        IChR_C1=(G_r*IChR_C2 + G_d1*IChR_O1 - k_1*IChR_C1)*dt+IChR_C1
        IChR_C2=(self.G_d2*IChR_O2 - (k_2 + G_r)*IChR_C2)*dt+IChR_C2          
        IChR_p=(((S0_tetha - IChR_p)/self.tau_ChR2 )*dt)+IChR_p

    
#Cell          

        E_Na=(R*T/F)*(log(Na_o/Na_i))
        INa=G_Na*pow(m,3)*h*j*(V-E_Na)
        m_inf=1.0/(pow((1.0+exp((-56.86-V)/9.03)),2))
        h_inf=1.0/(pow((1.0+exp((V+71.55)/7.43)),2))
        j_inf=h_inf
        alpha_m=1.0/(1.0+exp((-60.0-V)/5.0))
        beta_m=(0.1/(1.0+exp((V+35.0)/5.0)))+(0.1/(1.0+exp((V-50.0)/200.0)))
        tau_m=alpha_m*beta_m
        
        if(V>=-40.0):
        
            alpha_h=0.0
            beta_h=(0.77/(0.13*(1.0+exp(-1.0*(V+10.66)/11.1))))
            alpha_j=0.0
            beta_j=((0.6*exp(0.057*V))/(1.0+exp(-0.1*(V+32.0))))
        
        else:
        
            alpha_h=0.057*exp((-1.0*(V+80.0))/6.8)
            beta_h=(2.7*exp(0.079*V))+(3.1*pow(10,5)*exp(0.3485*V))
            alpha_j=(((-2.5428*pow(10,4)*exp(0.2444*V))+(-6.948*pow(10,-6)*exp(-0.04391*V)))*(V+37.78))/(1+exp(0.311*(V+79.23)))
            beta_j=(0.02424*exp(-0.01052*V))/(1.0+exp(-0.1378*(V+40.14)))
        
        tau_h=1.0/(alpha_h+beta_h)
        tau_j=1.0/(alpha_j+beta_j)
        m= m_inf-((m_inf-m)*exp(-dt/tau_m))
        h= h_inf-((h_inf-h)*exp(-dt/tau_h))
        j= j_inf-((j_inf-j)*exp(-dt/tau_j))
        VF_RT=(V*F)/(R*T)
        ICaL=(G_CaL*d*f*fCa)*4.0*(VF_RT*F)*(((Ca_i*exp(2.0*VF_RT))-(0.341*Ca_o))/(exp(2.0*VF_RT)-1.0))
        d_inf=1.0/(1.0+exp((-5.0-V)/7.5))
        alpha_d=(1.4/(1.0+exp((-35.0-V)/13.0)))+0.25
        beta_d=1.4/(1.0+exp((V+5.0)/5.0))
        gamma_d=1.0/(1.0+exp((50.0-V)/20.0))
        tau_d=alpha_d*beta_d+gamma_d
        f_inf=1.0/(1.0+exp((V+20.0)/7.0))
        tau_f=(1125.0*exp(-1.0*(pow((V+27.0),2))/240.0))+(165.0/(1.0+exp((25.0-V)/10.0)))+80.0
        alpha_fCa=1.0/(1.0+pow((Ca_i/0.000325),8))
        beta_fCa=0.1/(1.0+exp((Ca_i-0.0005)/0.0001))
        gamma_fCa=0.2/(1.0+exp((Ca_i-0.00075)/0.0008))
        fCa_inf=(alpha_fCa+beta_fCa+gamma_fCa+0.23)/1.46
        tau_fCa=2.0

        if(fCa_inf>fCa and V>-60.0):  
            
            klca = 0    
            
        else:
            
            klca = 1
            
        fCa=fCa_inf-((fCa_inf-fCa)*exp(-dt*klca/tau_fCa))        
        d= d_inf-((d_inf-d)*exp(-dt/tau_d))
        f=f_inf-((f_inf-f)*exp(-dt/tau_f))
        E_K=(R*T/F)*log(K_o/K_i)
        Ito=G_to_epi_M*r*s*(V-E_K)
        r_inf=(1.0/(1.0+exp((20.0-V)/6.0)))
        tau_r=(9.5*exp((-1.0*(pow((V+40.0),2)))/1800.0))+0.8
        EPIs_inf=1.0/(1.0+exp((V+20.0)/5.0))
        EPItau_s=(85.0*exp((-1.0*(pow((V+45.0),2)))/320.0))+(5.0/(1.0+exp((V-20.0)/5.0)))+3.0
        r=r_inf-((r_inf-r)*exp(-dt/tau_r))
        s=EPIs_inf-((EPIs_inf-s)*exp(-dt/EPItau_s))
        E_Ks=(R*T/F)*log((K_o+p_KNa*Na_o)/(K_i+p_KNa*Na_i))
        IKs=G_Ks_epi_endo*xs*xs*(V-E_Ks)
        xs_inf=1.0/(1.0+exp((-5.0-V)/14.0))
        alpha_xs=1100.0/(sqrt(1.0+exp((-10.0-V)/6.0)))
        beta_xs=1.0/(1.0+exp((V-60.0)/20.0))
        tau_xs=(alpha_xs)*(beta_xs)
        xs=xs_inf-((xs_inf-xs)*exp(-dt/tau_xs))
        IKr=G_Kr*(sqrt(K_o/5.4))*xr1*xr2*(V-E_K)
        xr1_inf=1.0/(1.0+exp((-26.0-V)/7.0))
        alpha_xr1=450.0/(1.0+exp((-45.0-V)/10.0))
        beta_xr1=6.0/(1.0+exp((V+30.0)/11.5))
        tau_xr1=alpha_xr1*beta_xr1
        xr1=xr1_inf-((xr1_inf-xr1)*exp(-dt/tau_xr1))
        xr2_inf=1.0/(1.0+exp((V+88.0)/24.0))
        alpha_xr2=3.0/(1.0+exp((-60.0-V)/20.0))
        beta_xr2=1.12/(1.0+exp((V-60.0)/20.0))
        tau_xr2=alpha_xr2*beta_xr2
        xr2=xr2_inf-((xr2_inf-xr2)*exp(-dt/tau_xr2))
        alpha_K1=0.1/(1.0+exp(0.06*(V-E_K-200.0)))
        beta_K1=((3.0*exp(0.0002*(V-E_K+100.0)))+(exp(0.1*(V-E_K-10.0))))/(1.0+exp(-0.5*(V-E_K)))
        xK1_inf=alpha_K1/(alpha_K1+beta_K1)
        IK1=G_K1*(sqrt(K_o/5.4))*xK1_inf*(V-E_K)
        INaCa=k_NaCa*(((exp(gama*VF_RT)*pow(Na_i,3)*Ca_o)-(exp((gama-1.0)*VF_RT)*pow(Na_o,3)*Ca_i*alpha))/((pow(K_mNai,3)+pow(Na_o,3))*(K_mCa+Ca_o)*(1.0+(k_sat*exp((gama-1.0)*VF_RT)))))
        INaK=P_NaK*((K_o*Na_i)/((K_o+K_mK)*(Na_i+K_mNa)*(1.0+(0.1245*exp(-0.1*VF_RT))+(0.0353*exp(-1.0*VF_RT)))))
        IpCa=G_pCa*(Ca_i/(K_pCa+Ca_i))
        IpK=G_pK*((V-E_K)/(1.0+exp((25.0-V)/5.98)))
        E_Ca=((R*T/F)/2.0)*log(Ca_o/Ca_i)
        IbNa=G_bNa*(V-E_Na)
        IbCa=G_bCa*(V-E_Ca)
        Ileak=V_leak*(Ca_sr-Ca_i)
        Iup=V_maxup/(1.0+(pow(K_up,2)/pow(Ca_i,2)))
        Irel=((a_rel*(pow(Ca_sr,2)/(pow(b_rel,2)+pow(Ca_sr,2))))+c_rel)*d*g
        
        if(Ca_i<0.00035):
                
            g_inf=1.0/(1.0+(pow(Ca_i,6)/pow(0.00035,6)))
                
        else:
               
            g_inf=1.0/(1.0+(pow(Ca_i,16)/pow(0.00035,16)))
                
        
        tau_g=2.0
        
        if(g_inf>g and V>-60.0): 
        
            kca = 0
            
        else:
            
            kca = 1
                  
        g=g_inf-((g_inf-g)*exp(-dt*kca/tau_g))
        Ca_itotal=Ca_i+(Ca_i*Buf_c)/(Ca_i+K_bufc)
        Ca_srbufsr=(Ca_sr*Buf_sr)/(Ca_sr+K_bufsr)
        Ca_srtotal=Ca_sr+Ca_srbufsr
        Ca_itotal=(((((-1.0*(ICaL+IbCa+IpCa-2.0*INaCa))/(2.0*V_c*F))*Cc)+Ileak-Iup+Irel)*dt) + Ca_itotal
        Ca_srtotal=(((V_c/V_SR)*(-Ileak+Iup-Irel))*dt)+Ca_srtotal     
        Ca_i=(-(K_bufc+Buf_c-Ca_itotal)+(sqrt((pow((K_bufc+Buf_c-Ca_itotal),2))+(4.0*Ca_itotal*K_bufc))))/2.0              
        Ca_sr=(-(K_bufsr+Buf_sr-Ca_srtotal)+(sqrt(pow((K_bufsr+Buf_sr-Ca_srtotal),2)+4.0*Ca_srtotal*K_bufsr)))/2.0      
        Na_i=((((-1.0*(INa+IbNa+(3.0*INaK)+(3.0*INaCa)))/(V_c*F))*Cc)*dt) + Na_i
        K_i=((((-1.0*(IK1+Ito+IKr+IKs-(2.0*INaK)+IpK))/(V_c*F))*Cc)*dt) + K_i  
        I_ion =INa+ICaL+Ito+IKs+IKr+IK1+INaCa+INaK+IpCa+IpK+IbNa+IbCa
        V1    = V - dt*(I_ion+stim+IChR2)/Cm

        self.Na_i  = Na_i
        self.Ca_i  = Ca_i
        self.K_i   = K_i
        self.Ca_sr = Ca_sr
        self.m     = m
        self.h     = h 
        self.j     = j
        self.d     = d
        self.f     = f
        self.fCa   = fCa
        self.r     = r
        self.s     = s
        self.xs    = xs
        self.xr1   = xr1
        self.xr2   = xr2
        self.g     = g
        self.V     = V1 

#        self.V     = -85.79647978
#        ChR2 vars
               
        self.IChR_O1 = IChR_O1
        self.IChR_O2 = IChR_O2
        self.IChR_C1 = IChR_C1
        self.IChR_C2 = IChR_C2
        self.IChR_p  = IChR_p
        self.IChR2   = IChR2
        self.I  = I