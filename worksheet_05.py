import numpy as np
import scipy.stats as stats
from scipy import fftpack, linalg, interpolate, signal
from scipy.optimize import curve_fit
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import time
import os



np.random.seed(0)

print("Q1:")
arr = np.random.randn(1000)
mean = arr.mean()
median = np.median(arr)
variance = arr.var(ddof=0)
scipy_mean = stats.tmean(arr)
scipy_median = np.median(arr)
scipy_var = arr.var()
print(f" mean={mean:.5f}, median={median:.5f}, var={variance:.5f}\n")




print("Q2:")
A = np.random.rand(4, 4)
A_fft2 = fftpack.fft2(A)
print("2D array:\n", A)
print("2D FFT (shape):", A_fft2.shape, "\n")




print("Q3:")
M = np.array([[4., 2., 1.],
              [2., 5., 3.],
              [1., 3., 6.]])
detM = linalg.det(M)
invM = linalg.inv(M)
eigvals, eigvecs = linalg.eig(M)
print("Matrix M:\n", M)
print("det(M) =", detM)
print("inv(M):\n", invM)
print("eigvals =", eigvals, "\n")




print("Q4:")
x = np.linspace(0, 10, 11)
y = np.sin(x) + 0.1 * np.random.randn(x.size)
interp_cubic = interpolate.interp1d(x, y, kind='cubic')
x_dense = np.linspace(0, 10, 200)
y_dense = interp_cubic(x_dense)
plt.figure()
plt.plot(x, y, 'o', label='data')
plt.plot(x_dense, y_dense, '-', label='cubic interp')
plt.legend()
plt.title("Q4: Cubic interpolation")
plt.savefig("Q4_interp.png")
plt.close()
print("Interpolation saved\n")




print("Q5:")
t = np.linspace(0, 10, 500)
sig = np.sin(2*np.pi*1.5*t) + 0.5*np.random.randn(t.size)
b, a = signal.butter(4, 0.2)
sig_filt = signal.filtfilt(b, a, sig)
plt.figure()
plt.plot(t, sig, label='noisy')
plt.plot(t, sig_filt, label='filtered')
plt.legend()
plt.title("Q5: Filtered signal")
plt.savefig("Q5_filter.png")
plt.close()
print("Signal filtered\n")




print("Q6:")
months = np.arange(1, 13)
products = ['P1', 'P2', 'P3', 'P4']
sales = np.random.randint(200, 2000, size=(12, 4))
total_sales_per_product = sales.sum(axis=0)
avg_sales_per_month = sales.mean(axis=1)
max_sale = sales.max()
min_sale = sales.min()
best_month_idx = sales.sum(axis=1).argmax() + 1
worst_month_idx = sales.sum(axis=1).argmin() + 1
print("Total sales per product:", dict(zip(products, total_sales_per_product)))
print("Best month:", best_month_idx, "Worst:", worst_month_idx)
plt.figure()
for i in range(4):
    plt.plot(months, sales[:, i], marker='o', label=products[i])
plt.legend()
plt.title("Q6 Sales")
plt.savefig("Q6_sales.png")
plt.close()
print("Q6 done\n")




print("Q7:")
students = ["Arin", "Aditya", "Chirag", "Gurleen", "Kunal"]
marks = np.array([
    [85, 78, 92, 88],
    [79, 82, 74, 90],
    [90, 85, 89, 92],
    [66, 75, 80, 78],
    [70, 68, 75, 85]
])
subjects = ["Math", "Physics", "Chemistry", "English"]
total_marks = marks.sum(axis=1)
avg_marks = marks.mean(axis=1)
subject_avg = marks.mean(axis=0)
top_idx = total_marks.argmax()
bottom_idx = total_marks.argmin()
passing_mark = 40
pass_counts = (marks >= passing_mark).all(axis=1).sum()
passing_percentage = pass_counts / len(students) * 100
print("Total:", dict(zip(students, total_marks)))
print("Top performer:", students[top_idx])
print("Passing %:", passing_percentage)
plt.figure()
plt.bar(subjects, subject_avg)
plt.title("Q7 Subject Averages")
plt.savefig("Q7_subject_avg.png")
plt.close()
print("Q7 done\n")




print("Q8:")
t_data = np.array([0, 1, 2, 3, 4, 5], float)
v_data = np.array([2, 3.1, 7.9, 18.2, 34.3, 56.2], float)
def quad(t, a, b, c): return a*t**2 + b*t + c
params, cov = curve_fit(quad, t_data, v_data)
a, b, c = params
print("Fitted parameters:", params)
t_fit = np.linspace(0, 5, 200)
v_fit = quad(t_fit, *params)
plt.figure()
plt.plot(t_data, v_data, 'o')
plt.plot(t_fit, v_fit, '-')
plt.title("Q8 Quadratic Fit")
plt.savefig("Q8_quad_fit.png")
plt.close()
print("Q8 done\n")




print("Q9:")
ind = np.arange(len(students))
width = 0.6
bottom = np.zeros(len(students))
plt.figure(figsize=(8,5))
for i in range(4):
    plt.bar(ind, marks[:, i], width, bottom=bottom, label=subjects[i])
    bottom += marks[:, i]
plt.xticks(ind, students)
plt.title("Q9 Marks Stacked")
plt.legend()
plt.savefig("Q9_marks_stacked.png")
plt.close()
print("Q9 done\n")




print("Q10:")
plt.figure()
plt.plot(t_data, v_data, 'o')
plt.plot(t_fit, v_fit, '-')
plt.xlabel("Time")
plt.ylabel("Velocity")
plt.title("Q10 Quadratic Fit Full")
plt.grid(True)
plt.savefig("Q10_quad_full.png")
plt.close()
print("Q10 done\n")




print("Q11:")
years = np.array([2000, 2005, 2010, 2015, 2020])
pop = np.array([50, 55, 70, 80, 90], float)
pearson_r, pval = stats.pearsonr(years, pop)
coeffs = np.polyfit(years, pop, 1)
m, c_lin = coeffs
pop_2008 = np.polyval(coeffs, 2008)
print("Pearson r:", pearson_r)
print("Pop 2008:", pop_2008)
years_dense = np.linspace(2000, 2020, 200)
plt.figure()
plt.plot(years, pop, 'o')
plt.plot(years_dense, np.polyval(coeffs, years_dense), '-')
plt.scatter([2008], [pop_2008], color='red')
plt.title("Q11 Population")
plt.savefig("Q11_pop.png")
plt.close()
print("Q11 done\n")




print("Q12:")
pcoeff = [3, -5, 2, -8]
roots = np.roots(pcoeff)
print("Roots:", roots)
x12 = np.linspace(-3, 3, 400)
p_x = np.polyval(pcoeff, x12)
plt.figure()
plt.plot(x12, p_x)
plt.axhline(0)
for r in roots:
    plt.plot(np.real(r), 0, 'ro')
plt.title("Q12 Polynomial")
plt.savefig("Q12_poly_roots.png")
plt.close()
print("Q12 done\n")




print("Q13:")
def generate_random_text_mb(path, mb_size):
    chunk = ("abcdefghijklmnopqrstuvwxyz0123456789 \n" * 1000).encode('utf-8')
    with open(path, 'wb') as f:
        written = 0
        while written < mb_size*1024*1024:
            f.write(chunk)
            written += len(chunk)

def convert_file_to_upper(src, dst):
    with open(src, 'r', encoding='utf-8', errors='ignore') as f_in:
        data = f_in.read()
    with open(dst, 'w', encoding='utf-8', errors='ignore') as f_out:
        f_out.write(data.upper())

sizes_mb = [1, 2]
timings = {}
for size in sizes_mb:
    fn = f"temp_{size}MB.txt"
    fn_up = f"temp_{size}MB_UPPER.txt"
    if not os.path.exists(fn):
        generate_random_text_mb(fn, size)
    t0 = time.perf_counter()
    convert_file_to_upper(fn, fn_up)
    dt = time.perf_counter() - t0
    timings[size] = dt
    print(f"{size}MB converted in {dt:.3f}s")
print("Q13 done\n")




print("Q14:")
def f14(x): return x**4 - 3*x**3 + 2
def d2f14(x): return 12*x**2 - 18*x
crit_roots = np.roots([4, -9, 0, 0])
crit_real = [np.real(r) for r in crit_roots if abs(np.imag(r)) < 1e-6]
mins = [r for r in crit_real if d2f14(r) > 0]
print("Local minima:", mins)
x14 = np.linspace(-2, 3, 400)
plt.figure()
plt.plot(x14, f14(x14))
for xm in mins:
    plt.plot(xm, f14(xm), 'ro')
plt.title("Q14 Minima")
plt.savefig("Q14_minima.png")
plt.close()
print("Q14 done\n")




print("Q15:")
zeta = 0.2
omega_n = 1.0
def damped_sys(y, t, zeta, omega_n):
    theta, theta_dot = y
    theta_ddot = -2*zeta*omega_n*theta_dot - (omega_n**2)*theta
    return [theta_dot, theta_ddot]
t15 = np.linspace(0, 20, 1000)
y0 = [1.0, 0.0]
sol = odeint(damped_sys, y0, t15, args=(zeta, omega_n))
theta = sol[:, 0]
max_disp = np.max(np.abs(theta))
t_max_disp = t15[np.argmax(np.abs(theta))]
print("Max displacement:", max_disp, "at time", t_max_disp)
plt.figure()
plt.plot(t15, theta)
plt.title("Q15 Damped Oscillator")
plt.savefig("Q15_damped.png")
plt.close()
print("Q15 done")
