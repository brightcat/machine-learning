package za.co.brightcat.machine.learning;

import za.co.brightcat.matrix.Matrix;

public interface Functions {
    Matrix h(Matrix X, Matrix theta);
    double cost(Matrix y, Matrix h);
    Matrix grad(double alpha, Matrix X, Matrix y, Matrix h);
}
