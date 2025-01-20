# 椭圆轨迹预测
由于要实现不同视角下观测并预测轨迹，所以不能再像之前一样使用圆来作为轨迹了。  
而我们已知圆的透视变换是椭圆，所以去拟合椭圆轨迹即可
## 椭圆的一般方程和标准方程的相互转换
本部分用sympy计算展开即可  
标准方程  
$$\frac{(x \cos \theta + y \sin \theta - h \cos \theta - k \sin \theta)^2}{a^2} + \frac{(-x \sin \theta + y \cos \theta + h \sin \theta - k \cos \theta)^2}{b^2} = 1$$  
展开，这里把sin和cos作为值来求 
$$
\begin{align}
x^{2} \left(\frac{sin_{val}^{2}}{b^{2}} + \frac{cos_{val}^{2}}{a^{2}}\right) \\ + x y \left(- \frac{2 cos_{val} sin_{val}}{b^{2}} + \frac{2 cos_{val} sin_{val}}{a^{2}}\right) \\+ x \left(\frac{2 cos_{val} k sin_{val}}{b^{2}} - \frac{2 h sin_{val}^{2}}{b^{2}} - \frac{2 cos_{val}^{2} h}{a^{2}} - \frac{2 cos_{val} k sin_{val}}{a^{2}}\right) \\+ y^{2} \left(\frac{cos_{val}^{2}}{b^{2}} + \frac{sin_{val}^{2}}{a^{2}}\right) \\+ y \left(- \frac{2 cos_{val}^{2} k}{b^{2}} + \frac{2 cos_{val} h sin_{val}}{b^{2}} - \frac{2 cos_{val} h sin_{val}}{a^{2}} - \frac{2 k sin_{val}^{2}}{a^{2}}\right) \\- 1 + \frac{cos_{val}^{2} k^{2}}{b^{2}} - \frac{2 cos_{val} h k sin_{val}}{b^{2}} + \frac{h^{2} sin_{val}^{2}}{b^{2}} + \frac{cos_{val}^{2} h^{2}}{a^{2}} + \frac{2 cos_{val} h k sin_{val}}{a^{2}} + \frac{k^{2} sin_{val}^{2}}{a^{2}}
\end{align}
$$
一般方程
$$ Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0 \quad \text{with} \quad B^2 - 4AC < 0 $$
因此令
$$
\begin{align}
A &=\frac{sin_{val}^{2}}{b^{2}} + \frac{cos_{val}^{2}}{a^{2}} \\  
B &= - \frac{2 cos_{val} sin_{val}}{b^{2}} + \frac{2 cos_{val} sin_{val}}{a^{2}}\\
C &= \frac{cos_{val}^{2}}{b^{2}} + \frac{sin_{val}^{2}}{a^{2}}\\
D &= \frac{2 cos_{val} k sin_{val}}{b^{2}} - \frac{2 h sin_{val}^{2}}{b^{2}} - \frac{2 cos_{val}^{2} h}{a^{2}} - \frac{2 cos_{val} k sin_{val}}{a^{2}}\\
E &= - \frac{2 cos_{val}^{2} k}{b^{2}} + \frac{2 cos_{val} h sin_{val}}{b^{2}} - \frac{2 cos_{val} h sin_{val}}{a^{2}} - \frac{2 k sin_{val}^{2}}{a^{2}}\\
F &= -1 + \frac{cos_{val}^{2} k^{2}}{b^{2}} - \frac{2 cos_{val} h k sin_{val}}{b^{2}} + \frac{h^{2} sin_{val}^{2}}{b^{2}} + \frac{cos_{val}^{2} h^{2}}{a^{2}} + \frac{2 cos_{val} h k sin_{val}}{a^{2}} + \frac{k^{2} sin_{val}^{2}}{a^{2}}\\
\end{align}
$$
