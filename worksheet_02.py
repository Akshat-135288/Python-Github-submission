# Question 1

# Initial list
L = [11, 12, 13, 14]
print("Original List:", L)

# (i) Add 50 and 60
L.append(50)
L.append(60)
print("(i) After adding 50 and 60:", L)

# (ii) Remove 11 and 13
L.remove(11)
L.remove(13)
print("(ii) After removing 11 and 13:", L)

# (iii) Sort ascending
L.sort()
print("(iii) Sorted ascending:", L)

# (iv) Sort descending
L.sort(reverse=True)
print("(iv) Sorted descending:", L)

# (v) Search for 13
if 13 in L:
    print("(v) 13 is present in the list")
else:
    print("(v) 13 is not present in the list")

# (vi) Count number of elements
print("(vi) Number of elements:", len(L))

# (vii) Sum of all elements
print("(vii) Sum of elements:", sum(L))

# (viii) Sum of odd numbers
odd_sum = sum(x for x in L if x % 2 != 0)
print("(viii) Sum of odd numbers:", odd_sum)

# (ix) Sum of even numbers
even_sum = sum(x for x in L if x % 2 == 0)
print("(ix) Sum of even numbers:", even_sum)

# (x) Sum of prime numbers
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

prime_sum = sum(x for x in L if is_prime(x))
print("(x) Sum of prime numbers:", prime_sum)

# (xi) Clear all elements
L.clear()
print("(xi) After clearing list:", L)

# (xii) Delete the list
del L
print("(xii) List deleted")





# Question 2
L = [1, 2, 3, 4, 5]
total = 0
for i in L:
    total += i
print("Sum of list elements:", total)



# Question 3
L = [1, 2, 3, 4, 5]
product = 1
for i in L:
    product *= i
print("Product of list elements:", product)



# Question 4
array = [[['*' for k in range(6)] for j in range(4)] for i in range(3)]

print("3D Array:")
for i in array:
    for j in i:
        print(j)
    print()



# Question 5
D = {1: 5.6, 2: 7.8, 3: 6.6, 4: 8.7, 5: 7.7}
print("Original Dictionary:", D)

D[8] = 8.8
print("(i) After adding key=8:", D)

if 2 in D:
    D.pop(2)
print("(ii) After removing key=2:", D)

if 6 in D:
    print("(iii) Key 6 is present")
else:
    print("(iii) Key 6 is not present")

print("(iv) Number of elements:", len(D))

total = 0
for value in D.values():
    total += value
print("(v) Sum of values:", total)

if 3 in D:
    D[3] = 7.1
print("(vi) After updating key=3:", D)

D.clear()
print("(vii) After clearing dictionary:", D)




# Question 6
S1 = {10, 20, 30, 40, 50, 60}
S2 = {40, 50, 60, 70, 80, 90}
print("Original S1:", S1)
print("Original S2:", S2)

S1.update([55, 66])
print("(i) After adding 55 and 66 in S1:", S1)

S1.discard(10)
S1.discard(30)
print("(ii) After removing 10 and 30 from S1:", S1)

if 40 in S1:
    print("(iii) 40 is present in S1")
else:
    print("(iii) 40 is not present in S1")

print("(iv) Union of S1 and S2:", S1.union(S2))
print("(v) Intersection of S1 and S2:", S1.intersection(S2))
print("(vi) S1 - S2:", S1.difference(S2))




# Question 7
import random
import string


# (i) Print 100 random strings whose length is between 6 and 8

print("100 Random Strings (length 6–8):")
for _ in range(100):
    length = random.randint(6, 8)              # choose random length
    rand_str = ''.join(random.choice(string.ascii_letters) for _ in range(length))
    print(rand_str)
print("\n" + "-"*50 + "\n")



# (ii) Print all prime numbers between 600 and 800

def is_prime(n):
    """Check if a number is prime"""
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

print("Prime numbers between 600 and 800:")
for num in range(600, 801):
    if is_prime(num):
        print(num, end=" ")
print("\n" + "-"*50 + "\n")



# (iii) Print all numbers between 100 and 1000 divisible by 7 and 9

print("Numbers between 100 and 1000 divisible by both 7 and 9:")
for num in range(100, 1001):
    if num % 7 == 0 and num % 9 == 0:   # means divisible by 63
        print(num, end=" ")




# Question 8
exam_st_date = (11, 12, 2014)
print("The examination will start from: %i / %i / %i" % exam_st_date)



# Question 9
numbers = [10, 23, 45, 66, 75, 90, 11, 22]
print("Numbers divisible by 5:")
for num in numbers:
    if num % 5 == 0:
        print(num)



# Question 10
n = int(input("Enter a number: "))
is_even = (n % 2 == 0)   # Boolean variable
if is_even:
    print("Even")
else:
    print("Odd")


# Question 11
text = "Emma is a good developer. Emma works at a company. Emma is smart."
count = text.count("Emma")
print("The substring 'Emma' appears", count, "times")





# Question 12
list1 = [10, 21, 4, 45, 66, 93]
list2 = [22, 33, 24, 55, 60, 71]

new_list = [x for x in list1 if x % 2 != 0] + [y for y in list2 if y % 2 == 0]
print("New List:", new_list)
