package za.co.brightcat.machine.learning;

import za.co.brightcat.matrix.Matrix;
import za.co.brightcat.matrix.MatrixOps;
import za.co.brightcat.matrix.MatrixUtil;

public class GradientDescent {
    final private MatrixOps ops;
    final private MatrixUtil util;

    public GradientDescent(MatrixOps ops, MatrixUtil util) {
        this.ops = ops;
        this.util = util;
    }
    
    public Matrix train(final Functions f, final Matrix X, final Matrix y, final double alpha, final int steps) {
        Matrix theta = util.zeros(X.n(), 1);
        for (int i = 0; i < steps; i++) {
            final Matrix h = f.h(X, theta);
            
            final double cost = f.cost(y, h);
            System.out.println("Cost: " + cost);
            
            final Matrix grad = f.grad(alpha, X, y, h);
            theta = ops.minus(theta, grad);
        }
        
        return theta;
    }
    
}
