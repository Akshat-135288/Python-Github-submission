import numpy as np
import math

# Q1
class Point:
    def __init__(self,x,y): self.x=x; self.y=y
    def dist(self,p): return ((self.x-p.x)**2+(self.y-p.y)**2)**0.5
    def mid(self,p): return ((self.x+p.x)/2,(self.y+p.y)/2)
def line_eq(A,B):
    m=(B.y-A.y)/(B.x-A.x)
    c=A.y-m*A.x
    return m,c
def reflect(C,A,B):
    m,c=line_eq(A,B)
    d=(C.x+m*(C.y-c))/(1+m*m)
    x=2*d-C.x; y=2*d*m-C.y+2*c
    return x,y

A=Point(1,2); B=Point(4,6); C=Point(3,5)
print(A.dist(B))
print(A.mid(B))
print(line_eq(A,B))
print(reflect(C,A,B))



# Q2
A=np.array([1,2]); B=np.array([3,4]); C=np.array([5,6])
print(A+B+C)
print(np.linalg.norm(A),np.linalg.norm(B),np.linalg.norm(C))
print(A.dot(B),A.dot(C),B.dot(C))
def ang(u,v): return math.degrees(math.acos(u.dot(v)/(np.linalg.norm(u)*np.linalg.norm(v))))
print(ang(A,B),ang(A,C),ang(B,C))
print((A.dot(B)/np.dot(B,B))*B)



# Q3
S=np.array([0,0]); E=np.array([4,4]); P=np.array([2,0])
SE=np.linalg.norm(E-S); print(SE)
t=max(0,min(1,np.dot(P-S,E-S)/np.dot(E-S,E-S)))
C=S+t*(E-S); print(C)
print(np.linalg.norm(P-C))



# Q4
a1,b1,c1=1,1,5
a2,b2,c2=2,-1,1
D=a1*b2-a2*b1
if D==0: print("Lines are parallel or coincident")
else:
    x=(c1*b2-b1*c2)/D
    y=(a1*c2-c1*a2)/D
    print(x,y)
