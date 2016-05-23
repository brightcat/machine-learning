package za.co.brightcat.machine.learning;

import za.co.brightcat.matrix.Matrix;
import za.co.brightcat.matrix.MatrixOps;
import za.co.brightcat.matrix.MatrixUtil;

public class GradientDescent {
    private int steps = 1500;
    private double alpha = 0.01;
    private MatrixOps ops;
    private MatrixUtil util;

    public GradientDescent(MatrixOps ops, MatrixUtil util) {
        this.ops = ops;
        this.util = util;
    }
    
    public Matrix train(Functions f, Matrix X, Matrix y) {
        Matrix theta = util.zeros(X.n(), 1);
        for (int i = 0; i < steps; i++) {
            final Matrix h = f.h(X, theta);
            
            final double cost = f.cost(y, h);
            System.out.println("Cost: " + cost);
            
            final Matrix grad = f.grad(X, y, h);
            final Matrix step = ops.map(grad, this::alpha);
            theta = ops.minus(theta, step);
        }
        
        return theta;
    }
    
    private double alpha(double x) {
        return alpha*x;
    }
}
