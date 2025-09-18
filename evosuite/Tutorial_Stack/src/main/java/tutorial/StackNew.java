package tutorial;

import java.util.EmptyStackException;

public class StackNew {
    private int[] values = new int[3];
    private int size = 0;

    void push(int x) {
        if (size >= values.length) {
            resize();
        }
        if (size < values.length) {
            values[size++] = x;
        }
    }

    int pop() {
        if (size > 0) {
            return values[--size];
        } else {
            throw new EmptyStackException();
        }
    }

    private void resize() {
        int[] tmp = new int[values.length * 2];
        for (int i = 0; i < values.length; i++) {
            tmp[i] = values[i];
        }
        values = tmp;
    }
}