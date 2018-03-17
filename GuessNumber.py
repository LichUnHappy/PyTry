import random

guesses_made = 0

name = input("Hello! What is your name?\n")

number = random.randint(1, 20)

print("Well, {0}, I am thinking of a number between 1 and 20.".format(name))

while guesses_made < 6:
    guess = int(input("Take a guess: "))
    guesses_made += 1

    if guess < number:
        print("Your guess is too low.")
    if guess > number:
        print("Your guess is too high.")
    if guess == number:
        break

if guess == number:
    print("Good job, {0}! You guss my number in {1} guesses!".format(name, guesses_made))
else:
    print("Shame on you.")

# 推荐在函数定义或类定义之间空两行，在类定义与第一个方法之间，或者需要进行语义分割的地方空一行
# 尽量调用者在上，被调用者在下。
# 二元运算符、布尔运算符左右有空格，逗号分号右边有空格，函数的默认参数两侧不要空格