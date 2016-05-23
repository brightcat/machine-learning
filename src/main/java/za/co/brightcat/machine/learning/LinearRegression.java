package za.co.brightcat.machine.learning;

import za.co.brightcat.matrix.Matrix;
import za.co.brightcat.matrix.MatrixOps;
import za.co.brightcat.matrix.MatrixUtil;

public class LinearRegression implements Functions {
    final private MatrixOps ops;
    final private MatrixUtil util;

    public LinearRegression(MatrixOps ops, MatrixUtil util) {
        this.ops = ops;
        this.util = util;
    }
    
    private double sqr(double x) {
        return x*x;
    }
    
    private double sum(double res, double v) {
        return res + v;
    }
    
    @Override
    public double cost(Matrix y, Matrix h) {
        final Matrix minus = ops.minus(h, y);
        final Matrix sqr = ops.map(minus, this::sqr);
        final double sum = ops.reduce(sqr, 0., this::sum);
        
        final int m = y.m();
        return 0.5 * sum / m;
    }

    @Override
    public Matrix grad(Matrix X, Matrix y, Matrix h) {
        final int m = y.m();
        final Matrix minus = ops.minus(h, y);
        final Matrix Xt = ops.transpose(X);
        final Matrix dot = ops.dot(Xt, minus);
        return ops.map(dot, x -> x / m);
    }

    @Override
    public Matrix h(Matrix X, Matrix theta) {
        return ops.dot(X, theta);
    }
    
}
