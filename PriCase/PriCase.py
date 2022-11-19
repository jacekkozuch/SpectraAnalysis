import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
import os
from scipy.optimize import minimize_scalar
import shutil

#################
####  Input  ####
input_pref = 'Spec'
num_of_specs = 503

wavenum_top = 2000
wavenum_bot = 1350
#################

def load_specs(input_pref,num_of_specs):

    in_set_wavenum = []
    in_set_spectra = []
    
    for a in range(0, num_of_specs):
        suffix = str(a+1).zfill(3)+'.txt'
        filename = input_pref + suffix
        x_temp,y_temp = np.loadtxt(filename).transpose()
        in_set_wavenum.append(x_temp)
        in_set_spectra.append(y_temp)
        
    return in_set_wavenum, in_set_spectra



def write_specs(set_wavenum, set_spectra,prefix):
    a = 0
    
    for waveum, spec in zip(set_wavenum, set_spectra):
        
        spectrum = np.array([waveum,spec]).transpose()
        
        suffix = str(a+1).zfill(3)+'.txt'
        np.savetxt(prefix+suffix, spectrum, fmt=['%.2f','%.10f'])
        a += 1
    
    
    
def prep_single_spectra(temp_set_wavenum, temp_set_spectra, wavenum_top, wavenum_bot):
    
    # This function does the following:
    # 1. Makes sure that all spectra run from high to low wavenumber
    # 2. Cuts spectra and interpolates intensities to integer wavenumbers
    #    - this solves compatibility issues between different spectrometers etc.
    #      since the wavenumber axes must be the same!
    
    in_set_wavenum = []
    in_set_spectra = []
    
    if temp_set_wavenum[0][0] > temp_set_wavenum[0][-1]:
        in_set_wavenum = temp_set_wavenum
        in_set_spectra = temp_set_spectra
    else:
        for wavenum, spectrum in zip(temp_set_wavenum, temp_set_spectra):
            in_set_wavenum.append(wavenum.reverse())
            in_set_spectra.append(spectrum.reverse())    

    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx
    
    def interpolate_specs(in_wavenum, in_spectra, wavenum_top, wavenum_bot):
        wavenum = []
        spectra = []
        wavenum_top = int(np.floor(wavenum_top))
        wavenum_bot = int(np.ceil(wavenum_bot))
        
        for a in range(wavenum_top, wavenum_bot-1, -1):
            wavenum.append(a)

            idx = find_nearest(in_wavenum, a)
            info = 'closest'
            
           
            if float(a) == in_wavenum[idx]:
                info = 'same'

            if info == 'same':
                value = in_spectra[idx]
                
            elif info == 'closest':
                
                if a > in_wavenum[idx]:
                    idx_top = idx-1
                    idx_bot = idx
                elif a < in_wavenum[idx]:
                    idx_top = idx
                    idx_bot = idx+1
  
                slope = (in_spectra[idx_top] - in_spectra[idx_bot])/(in_wavenum[idx_top] - in_wavenum[idx_bot])

                value = in_spectra[idx_top] + slope * (a - in_wavenum[idx_top])
            spectra.append(value)
            
        return wavenum, spectra
    
    new_set_wavenum = []
    new_set_spectra = []            
    for in_wavenum, in_spectra in zip(in_set_wavenum, in_set_spectra):
        temp_wavenum, temp_spectra = interpolate_specs(in_wavenum, in_spectra, wavenum_top, wavenum_bot)
        new_set_wavenum.append(temp_wavenum)
        new_set_spectra.append(temp_spectra)
        
    return new_set_wavenum, new_set_spectra



def show_all_specs(set_wavenum,set_spectra):
    for a in range(0, len(set_spectra)):
        plt.plot(set_wavenum[a], set_spectra[a], '-')
    plt.show()
    
def show_all_specs_2x1(set_wavenum1,set_spectra1,set_wavenum2,set_spectra2):
    for a in range(0, len(set_spectra1)):
        plt.subplot(1, 2, 1)
        plt.title('Before corrections')
        plt.xlabel('wavenumber / cm-1')
        plt.ylabel('Absorbance / OD')
        plt.plot(set_wavenum1[a], set_spectra1[a], '-')
    for a in range(0, len(set_spectra2)):
        plt.subplot(1, 2, 2)
        plt.title('After corrections')
        plt.xlabel('wavenumber / cm-1')
        plt.ylabel('Absorbance / OD')
        plt.plot(set_wavenum2[a], set_spectra2[a], '-')
    plt.subplots_adjust(wspace = 0.7)
    plt.show()
    
def show_scree_plot(rel_var,cum_var):


    fig, axs = plt.subplots(2)
    fig.suptitle('Scree Plots')
       
    axs[0].plot(rel_var, '-')
    axs[0].set_yscale('log')
    axs[1].plot(cum_var, '-')
    axs[1].set_yscale('log')

    plt.show()



def do_svd(specs):
    Vt, S, PC = np.linalg.svd(specs, full_matrices=False)
    
    return PC, S, Vt



def revert_svd(PC, S, Vt):
    
    PC = np.array(PC)
    
    PCxS = np.multiply(PC.transpose(),S).transpose()
    specs = np.matmul(Vt,PCxS)
    
    return specs


def correct_H2O_vapor(set_wavenum, set_PC, wavenum_top, wavenum_bot):
    init_corr_wavenum, init_corr_spec = load_specs('PriCase_H2Ovapor',1)
    corr_wavenum, corr_spec = prep_single_spectra(init_corr_wavenum, init_corr_spec, wavenum_top, wavenum_bot)
    def RMS(x, PC, corr_spec):
        resid = np.array(PC) - x * np.array(corr_spec)
        deriv = np.diff(resid)
        RMS = ((deriv - deriv.mean())**2).mean()
        return RMS
    new_set_PC = []
    for PC in set_PC:
        res = minimize_scalar(lambda x: RMS(x, PC, corr_spec))
        x = res.x
        old_RMS = RMS(0, PC, corr_spec)
        new_RMS = RMS(x, PC, corr_spec)

        if new_RMS < old_RMS:
            new_PC = np.array(PC) - x * np.array(corr_spec)
            new_set_PC.append(new_PC[0])
        else:
            new_set_PC.append(PC)                     
    return new_set_PC

def correct_CO2(set_wavenum, set_PC, wavenum_top, wavenum_bot):
    init_corr_wavenum, init_corr_spec = load_specs('PriCase_CO2',1)
    corr_wavenum, corr_spec = prep_single_spectra(init_corr_wavenum, init_corr_spec, wavenum_top, wavenum_bot)
    def RMS(x, PC, corr_spec):
        resid = np.array(PC) - x * np.array(corr_spec)
        deriv = np.diff(resid)
        RMS = ((deriv - deriv.mean())**2).mean()
        return RMS
    new_set_PC = []
    for PC in set_PC:
        res = minimize_scalar(lambda x: RMS(x, PC, corr_spec))
        x = res.x
        old_RMS = RMS(0, PC, corr_spec)
        new_RMS = RMS(x, PC, corr_spec)

        if new_RMS < old_RMS:
            new_PC = np.array(PC) - x * np.array(corr_spec)
            new_set_PC.append(new_PC[0])
        else:
            new_set_PC.append(PC)                     
    return new_set_PC

def save_cum_var(cum_var):
    cum_var_for_save = []
    count = 1
    for line in np.array(cum_var).transpose():
        temp = []
        temp.append(count)
        temp.append(line[0])
        temp.append(line[1])
        cum_var_for_save.append(temp)
        count += 1
    np.savetxt('00_cum_var.txt', cum_var_for_save, fmt=['%.d','%.5f','%.5f'])

######
# Main
decide = tk.messagebox.askyesno('PriCase - v1.0 - 2022/11/19', \
'This is a tool to correct large datasets of spectra using SVD.\n \
    \n \
1. Type into the header of the py-file the filename-prefix \n \
   (*001.txt,...), number of spectra, and wavenumber regions \n \
   to cut. \n \
   Click NO if you have not done that and then restart. \n \
   (note: your spectra will be reformatted to integer values \n \
   on x-axis with spacing of 1.) \n \
   Each round of corrections will be saved into separate \n \
   folder round001_*, etc. \n \
2. Basline correction - SVD will create Principle Components \n \
   (PCs), which can be corrected in an external program. \n \
   You can do also an automatic H2O vapor and CO2 \n \
   correction at this point, if desired. \n \
3. Component correction - PCs can be manipulated in an \n \
   external program (e.g. removing water) and a new and \n \
   the residual dataset of spectra will be saved. \n \
4. Combining PCs - you can combine chosen PCs and \n \
   reconstruct the associated dataset. \n \
   \n \
   Are you sure about this?')    
if decide == False:
    os.exit()




# Load set of spectra
init_set_wavenum, init_set_spectra = load_specs(input_pref,num_of_specs)

# Make generally compatible
init_set_wavenum, init_set_spectra = prep_single_spectra(init_set_wavenum, init_set_spectra, wavenum_top, wavenum_bot)
set_wavenum, set_spectra = init_set_wavenum, init_set_spectra
parent_path = os.getcwd()

# Start Baseline Corrections
round = 1
repeat = True
what_is_happening = 'baseline'
while repeat == True:
    
    rel_var = []
    cum_var = []
    new_dir = 'round' + str(round).zfill(3) + '_' + what_is_happening
    path = os.path.join(parent_path, new_dir)  
    try:
        os.mkdir(path)
    except:
        print("path already exists.")

    for file in os.listdir():
        if 'H2O' in file:
            shutil.copy(file, path)            
    for file in os.listdir():
        if 'CO2' in file:
            shutil.copy(file, path)   

    os.chdir(path)
    show_all_specs_2x1(set_wavenum,set_spectra,[],[])

    # Calculate SVD:                    
    PC, S, Vt = do_svd(set_spectra)
    rel_var.append(S**2/sum(S**2)*100)
    cum_var.append(np.cumsum(S**2/sum(S**2)*100))
    write_specs(set_wavenum, PC, 'out_PC')
    
    correct = tk.messagebox.askyesno('Round '+str(round), 'Autocorrect H2O vapor?')    
    if correct == True:
        print('correcting')
        PC = correct_H2O_vapor(set_wavenum, PC, wavenum_top, wavenum_bot)  
    correct = tk.messagebox.askyesno('Round '+str(round), 'Autocorrect CO2?')    
    if correct == True:
        print('correcting')
        PC = correct_CO2(set_wavenum, PC, wavenum_top, wavenum_bot)    
    write_specs(set_wavenum, PC, 'out_PCcorr')

    tk.messagebox.showinfo('Round '+str(round), 'Waiting for corrected PCs. Save them as in_PC001.txt etc. into the path %s.' %(new_dir))
    num_PCs = tk.simpledialog.askinteger('Round '+str(round), 'How many corrected PCs did you save?')
    
    # Loading new corrected PCs:
    init_PC_wavenum, init_PC = load_specs('in_PC',num_PCs)
    new_PC_wavenum, new_PC = prep_single_spectra(init_PC_wavenum, init_PC, wavenum_top, wavenum_bot)

    # Reconstructing dataset    
    new_S = S[0:num_PCs]
    new_Vt = ((Vt.transpose())[0:num_PCs]).transpose()
    new_spec = revert_svd(new_PC, new_S, new_Vt)
    show_all_specs_2x1(set_wavenum,set_spectra,set_wavenum,new_spec)
    write_specs(set_wavenum, new_spec, 'out_Spec')
    PC, S, Vt = do_svd(new_spec)
    rel_var.append(S**2/sum(S**2)*100)
    cum_var.append(np.cumsum(S**2/sum(S**2)*100))
    cum_var_for_save = []
    count = 1
    save_cum_var(cum_var)
                
    os.chdir(parent_path)    
    repeat = tk.messagebox.askyesno('Round '+str(round), 'Another round?')
    set_spectra = new_spec
    round += 1
                
                
                
                
repeat = tk.messagebox.askyesno('Round '+str(round), 'Do you want to substract spectra manually?')                
what_is_happening = 'manualcorrect'
while repeat == True:
    
    rel_var = []
    cum_var = []
    new_dir = 'round' + str(round).zfill(3) + '_' + what_is_happening
    path = os.path.join(parent_path, new_dir)  
    try:
        os.mkdir(path)
    except:
        print("path already exists.")

    os.chdir(path)
    show_all_specs(set_wavenum,set_spectra)

    # Calculate SVD:                    
    PC, S, Vt = do_svd(set_spectra)
    rel_var.append(S**2/sum(S**2)*100)
    cum_var.append(np.cumsum(S**2/sum(S**2)*100))
    write_specs(set_wavenum, PC, 'out_PC')
    
    tk.messagebox.showinfo('Round '+str(round), 'Wainting for corrected PCs. Save them as in_PC001.txt etc. into the path %s.' %(new_dir))
    num_PCs = tk.simpledialog.askinteger('Round '+str(round), 'How many corrected PCs did you save?')
    
    # Loading new corrected PCs:
    init_PC_wavenum, init_PC = load_specs('in_PC',num_PCs)
    new_PC_wavenum, new_PC = prep_single_spectra(init_PC_wavenum, init_PC, wavenum_top, wavenum_bot)

    res_PC = PC[0:num_PCs] - new_PC
    # Reconstructing dataset    
    new_S = S[0:num_PCs]
    new_Vt = ((Vt.transpose())[0:num_PCs]).transpose()
    new_spec = revert_svd(res_PC, new_S, new_Vt)
    show_all_specs(set_wavenum,new_spec)
    write_specs(set_wavenum, new_spec, 'out_Spec_res')

    # Reconstructing dataset    
    new_S = S[0:num_PCs]
    new_Vt = ((Vt.transpose())[0:num_PCs]).transpose()
    new_spec = revert_svd(new_PC, new_S, new_Vt)
    show_all_specs(set_wavenum,new_spec)
    write_specs(set_wavenum, new_spec, 'out_Spec')
    PC, S, Vt = do_svd(new_spec)
    rel_var.append(S**2/sum(S**2)*100)
    cum_var.append(np.cumsum(S**2/sum(S**2)*100))
    save_cum_var(cum_var)
               
    os.chdir(parent_path)    
    repeat = tk.messagebox.askyesno('Round '+str(round), 'Another round?')
    set_spectra = new_spec
    round += 1
    
    
    
repeat = tk.messagebox.askyesno('Round '+str(round), 'Do you want to split PC into groups?')                
what_is_happening = 'split'
if repeat == True:
    
    rel_var = []
    cum_var = []
    new_dir = 'round' + str(round).zfill(3) + '_' + what_is_happening
    path = os.path.join(parent_path, new_dir)  
    try:
        os.mkdir(path)
    except:
        print("path already exists.")

    os.chdir(path)
    show_all_specs(set_wavenum,set_spectra)

    # Calculate SVD:                    
    PC, S, Vt = do_svd(set_spectra)
    rel_var.append(S**2/sum(S**2)*100)
    cum_var.append(np.cumsum(S**2/sum(S**2)*100))
    write_specs(set_wavenum, PC, 'out_PC')

    while repeat == True:    
        which_PCs = tk.simpledialog.askstring('Round '+str(round), 'Which PCs should be combined (e.g 2, 3, 4)?')
    
        idx_PCs = []    
        for a in which_PCs.split(','):    
            idx_PCs.append(int(a))


        new_PC = []
        new_S  = []
        new_V = []
        for a in idx_PCs:
            new_PC.append(PC[a-1])
            new_S.append(S[a-1])
            new_V.append((Vt.transpose())[a-1])
        new_Vt = np.array(new_V).transpose()    
        # Reconstructing dataset    
        new_spec = revert_svd(new_PC, new_S, new_Vt)
        show_all_specs(set_wavenum,new_spec)
        write_specs(set_wavenum, new_spec, 'out_Spec_'+which_PCs+'_')
                    
 
        repeat = tk.messagebox.askyesno('Round '+str(round), 'Another combination?')

os.chdir(parent_path)   
tk.messagebox.showinfo('Done!', 'Good Job! Enjoy playing around with your spectra.')