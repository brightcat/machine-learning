package za.co.brightcat.machine.learning;

import java.io.IOException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import java.util.Random;
import za.co.brightcat.matrix.Matrix;
import za.co.brightcat.matrix.MatrixOps;
import za.co.brightcat.matrix.MatrixUtil;

public class LinearRegressionTest {
    private MatrixOps matrixOps() {
        return new MatrixOps();
    }
    private MatrixUtil matrixUtil() {
        return new MatrixUtil();
    }
    
    private GradientDescent gradientDescent(MatrixOps matrixOps, MatrixUtil matrixUtil) {
        return new GradientDescent(matrixOps, matrixUtil);
    }
    
    private double[][] getData() throws IOException {
        final String filename = this.getClass().getResource("lrdata.txt").getPath();//"/home/anton/coursera/ml/machine-learning-ex1/ex1/ex1data1.txt";
        List<String> lines = Files.readAllLines(Paths.get(filename));
        final double[][] data = lines.stream()
                .map(l -> l.split(","))
                .map(d -> {
                    final double[] r = new double[] { Double.parseDouble(d[0]), Double.parseDouble(d[1]) };
                    return r;
                })
                .toArray((int size) -> new double[size][2]);
        return data;
    }
    
    private Matrix x(double[][] in) {
        final double[][] data = new double[in.length][2];
        
        for (int i = 0; i < in.length; i++) {
            data[i][0] = 1.;
            data[i][1] = in[i][0];
        }
        
        return new Matrix(data);
    }
    
    private Matrix y(double[][] in) {
        final double[][] data = new double[in.length][1];
        
        for (int i = 0; i < in.length; i++) {
            data[i][0] = in[i][1];
        }
        
        return new Matrix(data);
    }
    
    public static void main(String[] args) throws IOException {
        final LinearRegressionTest ctx = new LinearRegressionTest();
    
        final MatrixOps matrixOps = ctx.matrixOps();
        final MatrixUtil matrixUtil = ctx.matrixUtil();
        
        final GradientDescent gradientDescent = ctx.gradientDescent(matrixOps, matrixUtil);
        
        final double[][] data = ctx.getData();
        final Matrix X = ctx.x(data);
        final Matrix y = ctx.y(data);
        
        final LinearRegression lr = new LinearRegression(matrixOps, matrixUtil);
        
        final Matrix t = gradientDescent.train(lr, X, y);
        
        matrixUtil.print(t);
        
        Matrix test1 = matrixUtil.create(new double[] { 1., 3.5 }, 1, 2);
        Matrix res1 = matrixOps.map(matrixOps.dot(test1, t), x -> x * 10000);
        matrixUtil.print(res1);
        Matrix test2 = matrixUtil.create(new double[] { 1., 7 }, 1, 2);
        Matrix res2 = matrixOps.map(matrixOps.dot(test2, t), x -> x * 10000);
        matrixUtil.print(res2);
        
        final double[][] randomData = ctx.randomData(10);
        final Matrix randomX = ctx.x(randomData);
        final Matrix randomy = ctx.y(randomData);
        
        final Matrix randomTheta = gradientDescent.train(lr, randomX, randomy);
        
        matrixUtil.print(randomTheta);
        
        final Matrix test3 = matrixUtil.create(new double[] { 1., 10 }, 1, 2);
        Matrix res3 = matrixOps.dot(test3, randomTheta);
        matrixUtil.print(res3);
    }
    
    private double f(double x) {
        return -5*x + 55;
    }
    
    private double[][] randomData(int size) {
        final double[][] data = new double[size][2];
        final Random r = new Random();
        for (int i = 0; i < size; i++) {
            double y = f(i);
            y += y * 0.1 * r.nextDouble();
            double x = i + (i*0.1*r.nextDouble());
            data[i][0] = x;
            data[i][1] = y;
        }
        
        return data;
    }
}
