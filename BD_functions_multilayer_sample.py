import numpy as np
import os
pi = 3.14159265358979323846264338327950
epsilon_0 = 8.854187817*10**(-12)
Planck = 6.62607015*10**(-34)
AVT_2 = 160.310615162421

solar_irradiation=np.loadtxt("./normals/solar_irradiation.txt")
eye_response=np.loadtxt("./normals/eye_response.txt")
tabxyz=np.loadtxt('./normals/xyz.txt')
d65spectra=np.loadtxt('./normals/D65.txt')

# 预加载文件数据
folder_path = "./opticalconstants"
file_data = {}

# 遍历文件夹，读取所有.nk文件
for filename in os.listdir(folder_path):
    if filename.endswith(".nk"):  # 只读取以.nk结尾的文件
        try:
            # 加载文件内容
            data = np.loadtxt(os.path.join(folder_path, filename))
            file_data[filename[:-3]] = data  # 去掉文件后缀作为键
        except OSError:
            print(f"Failed to read file: {filename}")

def create_n_values (x_range, filename):
    value = float(filename)
    n_values = x_range/x_range*value
    k_values = x_range*0.
    result = np.column_stack((x_range, n_values, k_values))
    return result

def np_interp (x_range, data):
    # 假设文件中的第一列是x值，第二列是y值，我们需要在x_range上插值
    x_values = data[:, 0]  # 原始x值
    y_values = data[:, 1]  # 对应的y值
    z_values = data[:, 2]  # 假设第三列也是要插值的值
    
    # 使用np.interp在x_range上进行插值
    y_interp = np.interp(x_range, x_values, y_values)
    z_interp = np.interp(x_range, x_values, z_values)
    
    # 将结果存储在二维数组中
    result = np.column_stack((x_range, y_interp, z_interp))
    return result

def load_txt_files(x_range, tabepsmat):
    n = len(tabepsmat)  # 获取输入列表的长度
    results = np.empty((n, len(x_range), 3))

    for i, filename in enumerate(tabepsmat):
        if isinstance(filename, float) or isinstance(filename, int):
            results[i] = create_n_values(x_range, filename)
        else:
            results[i] = np_interp (x_range, file_data[filename])
    return results

def format_time(seconds):
    if seconds < 1e-3:
        return f"{int(seconds * 1e6)} us"
    elif seconds < 1:
        return f"{int(seconds * 1e3)} ms"
    elif seconds < 60:
        return f"{int(seconds)} s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02}:{seconds:02}"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02}:{minutes:02}:{seconds:02}"
    
def comp_eps(tab, lambdas):
    nmed = tab.shape[0]
    nlamb = lambdas.shape[0]
    
    ceps = np.zeros((nmed, nlamb), dtype=np.complex128)
    
    for imed in range(nmed):
        # nr = linear_interp(lambdas, tab[imed, :, 0] / 1000, tab[imed, :, 1])
        # ni = linear_interp(lambdas, tab[imed, :, 0] / 1000, tab[imed, :, 2])
        nr = np.interp(lambdas, tab[imed, :, 0] / 1000, tab[imed, :, 1])
        ni = np.interp(lambdas, tab[imed, :, 0] / 1000, tab[imed, :, 2])
        
        ceps[imed, :] = nr**2 - ni**2 + 2j * nr * ni
    return ceps

def ampE(nmed, zint, epsmat, kk, kz):
    # 初始化结果数组，直接操作多维数组，避免循环
    nlambda = kk.shape[1]
    AA = np.zeros((2, nmed, nlambda), dtype=np.complex128)
    
    # 初始化 alpha, beta, gama, delta
    alpha = np.zeros((nmed, nlambda), dtype=np.complex128)
    beta = np.zeros((nmed, nlambda), dtype=np.complex128)
    gama = np.zeros((nmed, nlambda), dtype=np.complex128)
    delta = np.zeros((nmed, nlambda), dtype=np.complex128)
    
    for i in range(nmed - 1):
        c1 = np.exp(1j * kz[i] * zint[i])
        c2 = np.exp(1j * kz[i + 1] * zint[i])
        
        alpha[i] = (kk[i] / kk[i + 1]) * (epsmat[i] * kz[i + 1] + epsmat[i + 1] * kz[i]) / (2 * epsmat[i] * kz[i]) * c2 / c1
        beta[i] = (kk[i] / kk[i + 1]) * (epsmat[i] * kz[i + 1] - epsmat[i + 1] * kz[i]) / (2 * epsmat[i] * kz[i]) / c1 / c2
        gama[i] = beta[i] * c1 * c2 * c1 * c2
        delta[i] = alpha[i] * c1 / c2 * c1 / c2
        
    tab1, tab2, tab3, tab4 = alpha[nmed - 2], beta[nmed - 2], gama[nmed - 2], delta[nmed - 2]
    for i in range(nmed - 2):
        tabn1 = alpha[nmed - 3 - i] * tab1 + beta[nmed - 3 - i] * tab3
        tabn2 = alpha[nmed - 3 - i] * tab2 + beta[nmed - 3 - i] * tab4
        tabn3 = gama[nmed - 3 - i] * tab1 + delta[nmed - 3 - i] * tab3
        tabn4 = gama[nmed - 3 - i] * tab2 + delta[nmed - 3 - i] * tab4
        tab1, tab2, tab3, tab4 = tabn1, tabn2, tabn3, tabn4
            
    AA[0, nmed - 1] = 1. / tab1
    AA[1, nmed - 1] = 0.
    
    for i in range(nmed - 1):
        AA[0, nmed - 2 - i] = alpha[nmed - 2 - i] * AA[0, nmed - 1 - i] + beta[nmed - 2 - i] * AA[1, nmed - 1 - i]
        AA[1, nmed - 2 - i] = gama[nmed - 2 - i] * AA[0, nmed - 1 - i] + delta[nmed - 2 - i] * AA[1, nmed - 1 - i]
    return AA

def calculate_A (tablamb,epsmat,nmed,ipm6,zint,AA,kk,kz):
    # valr=np.abs(AA[1,0,:])**2*np.real(kz[0,:]/kz[0,:])*100
    # valt=np.abs(AA[0,nmed-1,:])**2*np.real(kz[nmed-1,:]/kz[0,:])*100

    aphase=np.angle(AA[0,ipm6,:]*np.conjugate(AA[1,ipm6,:]))
    kzre=np.real(kz[ipm6])
    kzim=np.imag(kz[ipm6])
    aa=zint[ipm6-1]
    bb=zint[ipm6]

    # 初始化 fap 和 fam 数组
    fap = np.empty_like(kzim)
    fam = np.empty_like(kzim)

    # 使用 NumPy 的条件推导式来处理 kzim
    non_zero_kzim = kzim != 0

    # 对于 kzim != 0 的情况
    fap[non_zero_kzim] = (np.exp(-2 * kzim[non_zero_kzim] * aa) - np.exp(-2 * kzim[non_zero_kzim] * bb)) / (2 * kzim[non_zero_kzim])
    fam[non_zero_kzim] = (np.exp(2 * kzim[non_zero_kzim] * bb) - np.exp(2 * kzim[non_zero_kzim] * aa)) / (2 * kzim[non_zero_kzim])

    # 对于 kzim == 0 的情况
    fap[~non_zero_kzim] = bb - aa
    fam[~non_zero_kzim] = bb - aa

    epsl=epsmat[ipm6,:]
    nin=np.real(np.sqrt(epsmat[0,:]))
    vabslayer=100*2*np.pi/tablamb/nin*np.imag(epsl)*(np.abs(AA[0,ipm6])**2*fap+np.abs(AA[1,ipm6])**2*fam+np.abs(AA[0,ipm6])*np.abs(AA[1,ipm6])/kzre*(np.sin(2*kzre*bb+aphase)-np.sin(2*kzre*aa+aphase)))
    return vabslayer

def calcule_absorption (tablamb, vabslayer):
    tablamb = tablamb * 1000
    
    result_absorption = np.column_stack((tablamb,vabslayer))
    
    return result_absorption

def initialize_parameters(tab, ipm6, elayer, lambmin, lambmax, dlamb, incident_angle):
    nmed = tab.shape[0]
    
    thetai = incident_angle * np.pi / 4
    
    lambmax = lambmax / 1000
    lambmin = lambmin / 1000
    dlamb = dlamb / 1000
    tablamb = np.arange(lambmin, lambmax + 0.1 * dlamb, dlamb)
    
    elayer = elayer / 1000
    
    zint = np.zeros(len(elayer) + 1, dtype=float)
    zint[1:] = np.cumsum(elayer)
    
    sinus = np.sin(thetai)
    
    return nmed, tablamb, zint, sinus

def main (tab, ipm6, elayer, lambmin, lambmax, dlamb, incident_angle):
    nmed, tablamb, zint, sinus = initialize_parameters(tab, ipm6, elayer, lambmin, lambmax, dlamb, incident_angle)

    # tab 是 (7, 80, 3) 的数组，nmed 是材料的个数，tablamb是波长范围
    epsmat = comp_eps(tab, tablamb)

    k0=2*np.pi/tablamb
    kx=np.sqrt(epsmat[0,:])*k0*sinus
    kk=np.sqrt(epsmat)*k0
    kz=np.sqrt((kk**2-kx**2))

    AA = ampE(nmed, zint, epsmat, kk, kz)
    vabslayer = calculate_A (tablamb,epsmat,nmed,ipm6,zint,AA,kk,kz)

    result_absorption = calcule_absorption (tablamb, vabslayer)
    
    return result_absorption

    
