import tkinter as tk
import math



# ================= Q1 =================
root = tk.Tk()
root.title("Robot Control Panel")
root.geometry("500x400")
root.configure(bg="yellow")
root.mainloop()



# ================= Q2 =================
root = tk.Tk()
c = tk.Canvas(root, width=300, height=300)
c.pack()
c.create_oval(98,98,102,102, fill="black")
root.mainloop()



# ================= Q3 =================
root = tk.Tk()
c = tk.Canvas(root, width=300, height=200)
c.pack()
points = [(20,50),(100,120),(180,90),(250,150)]
flat = [x for p in points for x in p]
c.create_line(flat, fill="blue", width=3)
root.mainloop()




# ================= Q4 =================
root = tk.Tk()
c = tk.Canvas(root, width=400, height=200)
c.pack()
dot = c.create_oval(10,90,20,100, fill="red")
def move():
    c.move(dot,5,0)
    root.after(100,move)
move()
root.mainloop()




# ================= Q5 =================
root = tk.Tk()
c = tk.Canvas(root, width=400, height=300)
c.pack()
c.create_rectangle(150,120,250,180, fill="gray")
c.create_oval(150,180,180,210, fill="black")
c.create_oval(220,180,250,210, fill="black")
root.mainloop()




# ================= Q6 =================
root = tk.Tk()
c = tk.Canvas(root, width=400, height=300)
c.pack()
robot = c.create_oval(190,140,210,160, fill="blue")
def move(dx,dy): c.move(robot,dx,dy)
tk.Button(root,text="Up",command=lambda:move(0,-10)).pack()
tk.Button(root,text="Down",command=lambda:move(0,10)).pack()
tk.Button(root,text="Left",command=lambda:move(-10,0)).pack()
tk.Button(root,text="Right",command=lambda:move(10,0)).pack()
root.mainloop()




# ================= Q7 =================
root = tk.Tk()
c = tk.Canvas(root, width=400, height=300)
c.pack()
ball = c.create_oval(185,135,215,165, fill="red")
dx,dy = 3,3
def bounce():
    global dx,dy
    c.move(ball,dx,dy)
    x1,y1,x2,y2 = c.coords(ball)
    if x1<=0 or x2>=400: dx=-dx
    if y1<=0 or y2>=300: dy=-dy
    root.after(30,bounce)
bounce()
root.mainloop()




# ================= Q8 =================
root = tk.Tk()
c = tk.Canvas(root, width=500, height=300)
c.pack()
robot = c.create_oval(45,195,55,205, fill="green")
def move():
    c.move(robot,5,0)
    if c.coords(robot)[2] < 450:
        root.after(50,move)
move()
root.mainloop()




# ================= Q9 =================
root = tk.Tk()
c = tk.Canvas(root, width=600, height=400)
c.pack()
A=(150,300); D=(400,300)
L2=120; theta=math.radians(30)
B=(A[0]+L2*math.cos(theta),A[1]-L2*math.sin(theta))
c.create_line(A,D,width=3)
c.create_line(A,B,fill="red",width=3)
for p in [A,B,D]:
    c.create_oval(p[0]-4,p[1]-4,p[0]+4,p[1]+4,fill="black")
root.mainloop()




# ================= Q10 =================
root = tk.Tk()
c = tk.Canvas(root, width=500, height=400)
c.pack()
robot = c.create_oval(245,195,255,205, fill="blue")
path=[]
def move(dx,dy):
    c.move(robot,dx,dy)
    x1,y1,x2,y2 = c.coords(robot)
    path.append(c.create_line(x1,y1,x2,y2))
def reset():
    for p in path: c.delete(p)
root.bind("<Up>",lambda e:move(0,-5))
root.bind("<Down>",lambda e:move(0,5))
root.bind("<Left>",lambda e:move(-5,0))
root.bind("<Right>",lambda e:move(5,0))
tk.Button(root,text="RESET",command=reset).pack()
root.mainloop()
