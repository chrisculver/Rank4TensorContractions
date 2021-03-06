import matplotlib.pyplot as plt

plt.rcParams['text.usetex']=True
plt.rcParams['text.latex.preamble'] = '\\usepackage{amsmath}'

f=open('contract_times.dat','r')

nvec=[]
timeUS=[]
memory=[]
for line in f.readlines():
  l=line.split(' ')
  n=int(l[0])
  nvec.append(n)
  timeUS.append(int(l[1].strip('\n')))
  memory.append(float(4*n*n*n*n*2*8)/(1e+9))
print(nvec)
print(timeUS)
print(memory)

fig,ax1=plt.subplots()

pts1=ax1.plot(nvec, timeUS, marker='o', markerfacecolor="None", markeredgecolor='C0', linestyle='None', label='Time')


ax2=ax1.twinx()
pts2=ax2.plot(nvec, memory, marker='s', markerfacecolor="None", markeredgecolor='C1', linestyle='None', label='Memory')

pts=pts1+pts2
ptLabels=[p.get_label() for p in pts]
ax1.legend(pts, ptLabels, loc=0)

ax1.set_yscale('log')

ax1.set_ylabel('$t\\,\\left[us\\right]$')
ax1.set_xlabel('$N_{\\text{vec}}N_{s}$')
ax2.set_ylabel('$\\text{Memory}\\, \\left[ \\text{GB} \\right]$')

plt.title('$A_{abij}B_{jicd}C_{abkl}D_{klcd}$')
#plt.savefig('contract_time.pdf')
plt.savefig('contract_time.svg')
plt.show()
