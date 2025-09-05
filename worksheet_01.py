# Question 1
print("Twinkle, twinkle, little star,\n"
      "\tHow I wonder what you are!\n"
      "\t\tUp above the world so high,\n"
      "\t\tLike a diamond in the sky.\n"
      "Twinkle, twinkle, little star,\n"
      "\tHow I wonder what you are")



# Question 2
first = input("Enter your first name: ")
last = input("Enter your last name: ")
print(last, first)


# Question 3
import math
r = float(input("Enter radius: "))
area = math.pi * r * r
print("Area of circle:", area)


# Question 4
color_list = ["Red", "Green", "White", "Black"]
print(color_list[0], color_list[-1])


# Question 5
n = int(input("Enter a number: "))
result = n + int(str(n)*2) + int(str(n)*3)
print(result)


# Question 6
values = input("Enter numbers separated by commas: ")
list_values = values.split(",")
tuple_values = tuple(list_values)
print("List:", list_values)
print("Tuple:", tuple_values)


# Question 7
c = float(input("Enter Celsius: "))
f = (c * 9/5) + 32
print("Fahrenheit:", f)



# Question 8
a = int(input("Enter first number: "))
b = int(input("Enter second number: "))

a = a + b
b = a - b
a = a - b

print("After swapping: a =", a, ", b =", b)




# Question 9
n = int(input("Enter a number: "))
if n % 2 == 0:
    print("Even")
else:
    print("Odd")



# Question 10
year = int(input("Enter a year: "))
if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
    print("Leap Year")
else:
    print("Not a Leap Year")



# Question 11
import math

# Input coordinates
x1, y1 = map(float, input("Enter x1 and y1: ").split())
x2, y2 = map(float, input("Enter x2 and y2: ").split())

# Euclidean distance formula
distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

print("Euclidean Distance:", distance)





# Question 12
a = int(input("Enter first angle: "))
b = int(input("Enter second angle: "))
c = int(input("Enter third angle: "))

if a + b + c == 180:
    print("Valid Triangle")
else:
    print("Not a Triangle")



# Question 13

p = float(input("Enter principal: "))
r = float(input("Enter rate: "))
t = float(input("Enter time (in years): "))

ci = p * (pow((1 + r / 100), t)) - p
print("Compound Interest:", ci)




# Question 14
n = int(input("Enter a number: "))
if n > 1:
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            print("Not Prime")
            break
    else:
        print("Prime")
else:
    print("Not Prime")





# Question 15
N = int(input("Enter a number: "))
total = sum(i*i for i in range(1, N+1))
print("Sum of squares:", total)
