# Project 1: Tic Tac Toe
board=[" "]*9
def print_b(): print(board[0:3],board[3:6],board[6:9])
def win(b,p):
    c=[[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]]
    return any(b[x]==b[y]==b[z]==p for x,y,z in c)
def tie(b): return " " not in b
def inp(p):
    while True:
        x=int(input(f"{p} pos:"))-1
        if 0<=x<9 and board[x]==" ": return x
def game():
    p="X"
    while True:
        print_b()
        x=inp(p); board[x]=p
        if win(board,p): print_b(); print(p,"wins"); return
        if tie(board): print_b(); print("Tie"); return
        p="O" if p=="X" else "X"
# game()

# Project 2: To-Do List
tasks=[]
def add(): tasks.append(input("Task:"))
def view(): [print(i,t) for i,t in enumerate(tasks)]
def delete(): i=int(input("Index:")); print("Invalid" if i>=len(tasks) else tasks.pop(i))
def todo():
    while True:
        c=input("1-add 2-view 3-del 4-exit:")
        if c=="1": add()
        elif c=="2": view()
        elif c=="3": delete()
        else: break
# todo()

# Project 3: A* Robot Path
import numpy as np, matplotlib.pyplot as plt, heapq, pandas as pd
def astar(g,s,e):
    r,c=len(g),len(g[0])
    h=lambda x,y:abs(x-e[0])+abs(y-e[1])
    pq=[(0,*s,None)]; vis=set(); prev={}
    while pq:
        f,x,y,p=heapq.heappop(pq)
        if (x,y) in vis: continue
        vis.add((x,y)); prev[(x,y)]=p
        if (x,y)==e:
            path=[e]; 
            while prev[path[-1]]: path.append(prev[path[-1]])
            return path[::-1]
        for dx,dy in[(1,0),(-1,0),(0,1),(0,-1)]:
            nx,ny=x+dx,y+dy
            if 0<=nx<r and 0<=ny<c and g[nx][ny]==0:
                heapq.heappush(pq,(h(nx,ny),nx,ny,(x,y)))
    return None

def run():
    R,C=map(int,input("rows cols:").split())
    g=np.zeros((R,C),int)
    k=int(input("obs count:"))
    for _ in range(k):
        x,y=map(int,input().split()); g[x][y]=1
    s=tuple(map(int,input("start:").split()))
    e=tuple(map(int,input("end:").split()))
    print(pd.DataFrame(g))
    path=astar(g,s,e)
    if not path: print("No path"); return
    x=[i[0] for i in path]; y=[i[1] for i in path]
    plt.imshow(g,cmap="gray"); plt.plot(y,x,"r"); 
    plt.scatter(s[1],s[0],c="g"); plt.scatter(e[1],e[0],c="b")
    plt.show()
# run()
