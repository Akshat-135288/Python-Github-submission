# Question 1
def difference_17(n):
    if n > 17:
        return 2 * abs(n - 17)
    else:
        return 17 - n

print("1.", difference_17(22))
print("1.", difference_17(14))





# Question 2
def test_range(n):
    return (100 <= n <= 1000) or (n == 2000)

print("2.", test_range(500))
print("2.", test_range(2000))





# Question 3
def reverse_string(s):
    return s[::-1]

print("3.", reverse_string("Python"))





# Question 4
def count_case(s):
    upper = sum(1 for c in s if c.isupper())
    lower = sum(1 for c in s if c.islower())
    return upper, lower

print("4.", count_case("Hello World"))




# Question 5
def unique_list(lst):
    return list(set(lst))

print("5.", unique_list([1, 2, 3, 2, 4, 1, 5]))




# Question 6
def even_numbers(lst):
    return [x for x in lst if x % 2 == 0]

print("6.", even_numbers([1, 2, 3, 4, 5, 6, 7, 8, 9]))


# Question 7
def outer_function(a):
    def inner_function(b):
        return b * b
    return inner_function(a)

print("7.", outer_function(5))




# Question 8
def student(name, age):
    return f"Name: {name}, Age: {age}"

print("8.", student.__code__.co_varnames)  # shows argument names


# Question 9
class Student:
    def __init__(self, student_id, student_name, student_class):
        self.student_id = student_id
        self.student_name = student_name
        self.student_class = student_class

    def display(self):
        print("ID:", self.student_id)
        print("Name:", self.student_name)
        print("Class:", self.student_class)

s = Student(101, "Akshat", "CSE")
print("9.")
s.display()





# Question 10
class Student2:
    def __init__(self, student_id, student_name, student_class):
        self.student_id = student_id
        self.student_name = student_name
        self.student_class = student_class

student1 = Student2(101, "Alice", "CSE")
student2 = Student2(102, "Bob", "ECE")

print("10.")
print(f"Student1 → ID: {student1.student_id}, Name: {student1.student_name}, Class: {student1.student_class}")
print(f"Student2 → ID: {student2.student_id}, Name: {student2.student_name}, Class: {student2.student_class}")





# Question 11
import math
class Circle:
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return math.pi * self.radius ** 2

    def perimeter(self):
        return 2 * math.pi * self.radius

c = Circle(7)
print("11. Area:", c.area())
print("11. Perimeter:", c.perimeter())





# Question 12
class StringHandler:
    def get_String(self):
        self.s = input("Enter a string: ")

    def print_String(self):
        print(self.s.upper())

sh = StringHandler()
# Uncomment to test with input:
# sh.get_String()
# sh.print_String()
