import matplotlib.pyplot as plt
from scipy import stats

# Expanded dataset (more points)
x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
y = [3, 5, 5, 6, 7, 8, 9, 11, 12, 14]

# Perform linear regression
slope, intercept, r, p, std_err = stats.linregress(x, y)
print("slope: ", slope)
print("intercept: ", intercept)

# Linear function definition
def myfunc(x_val):
    return slope * x_val + intercept

# Map predictions
mymodel = list(map(myfunc, x))

# Plotting with enhancements
plt.figure(figsize=(8, 5))
plt.scatter(x, y, color='blue', label='Data Points')
plt.plot(x, mymodel, color='red', label='Linear Fit')

# Add Labels, Title, Legend, and Grid
plt.xlabel("X Axis")
plt.ylabel("Y Axis")
plt.title("Linear Regression with Stats")
plt.legend()
plt.grid(True)

plt.show()