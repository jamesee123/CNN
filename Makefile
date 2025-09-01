make build:
	gcc $(wildcard src/*/*.c) $(wildcard examples/main.c) -Iinclude -o builds/out -lm