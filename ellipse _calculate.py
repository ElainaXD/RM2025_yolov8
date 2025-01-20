from sympy import symbols, cos, sin, expand, simplify, collect, latex,Eq,solve

# 定义变量
x, y, h, k, a, b, theta, sin_val, cos_val, A,B,C,D,E,F = symbols('x y h k a b theta sin_val cos_val A B C D E F')

# 定义椭圆的参数方程，使用 sin_val 和 cos_val 代替 sin(theta) 和 cos(theta)
X = x * cos_val + y * sin_val - h * cos_val - k * sin_val
Y = -x * sin_val + y * cos_val + h * sin_val - k * cos_val

# 将参数方程代入椭圆的标准方程
equation = (X**2 / a**2) + (Y**2 / b**2) - 1

# 展开方程
expanded_equation = expand(equation)

# 合并同类项
collected_equation = collect(expanded_equation, [x**2, x*y, y**2, x, y])

# 提取系数
A_expr = collected_equation.coeff(x**2)
B_expr = collected_equation.coeff(x*y)
C_expr = collected_equation.coeff(y**2)
# 在提取x的系数时，将y设为0
D_expr = collected_equation.subs(y, 0).coeff(x)
# 在提取y的系数时，将x设为0
E_expr = collected_equation.subs(x, 0).coeff(y)
# 提取常数项需要将x和y设为0
F_expr = collected_equation.subs({x: 0, y: 0})

eq1=Eq(A,A_expr)
eq2=Eq(B,B_expr)
eq3=Eq(C,C_expr)
eq4=Eq(D,D_expr)
eq5=Eq(E,E_expr)
eq6=Eq(F,F_expr)
# 添加三角函数的约束条件
eq7 = Eq(sin_val**2 + cos_val**2, 1)
# 尝试求解方程组
solution = solve((eq1, eq2, eq3, eq4, eq5, eq6, eq7), (h, k, a, b, sin_val, cos_val), dict=True)

# 打印合并后的方程
print(collected_equation)

# 如果需要 LaTeX 格式的输出，将 sin_val 和 cos_val 替换回 sin(theta) 和 cos(theta)
latex_output = latex(collected_equation).replace('sin_val', r'\sin(\theta)').replace('cos_val', r'\cos(\theta)')
print(latex_output)

# 输出每个变量对应的式子
print(f"A 对应的式子: {latex(A_expr)}")
print(f"B 对应的式子: {latex(B_expr)}")
print(f"C 对应的式子: {latex(C_expr)}")
print(f"D 对应的式子: {latex(D_expr)}")
print(f"E 对应的式子: {latex(E_expr)}")
print(f"F 对应的式子: {latex(F_expr)}")

print(solution)