namespace NeuralNetwork;

public class NeuralNetwork
{
    private double[,] _weights;

    private enum OPERATION
    {
        Multiply,
        Add,
        Subtract
    }

    public NeuralNetwork()
    {
        var randomNumber = new Random(1);
        var numberOfInputNodes = 3;
        var numberOfOutputNodes = 1;
        _weights = new double[numberOfInputNodes, numberOfOutputNodes];
        for (var i = 0; i < numberOfInputNodes; i++)
        {
            for (var j = 0; j < numberOfOutputNodes; j++)
            {
                _weights[i, j] = 2 * randomNumber.NextDouble() - 1;
            }
        }
    }

    private double[,] Activate(double[,] matrix, bool isDerivative)
    {
        var numberOfRows = matrix.GetLength(0);
        var numberOfColumns = matrix.GetLength(1);
        var result = new double[numberOfRows, numberOfColumns];

        for (var row = 0; row < numberOfRows; row++)
        {
            for (var column = 0; column < numberOfColumns; column++)
            {
                var sigmoidOutput = result[row, column] = 1 / (1 + Math.Exp(-matrix[row, column]));
                var derivativeSigmoidOutput = result[row, column] = matrix[row, column] * (1 - matrix[row, column]);
                result[row, column] = isDerivative ? derivativeSigmoidOutput : sigmoidOutput;
            }
        }

        return result;
    }

    public void Train(double[,] trainingInputs, double[,] trainingOutputs, int numberOfIterations)
    {
        for (var iteration = 0; iteration < numberOfIterations; iteration++)
        {
            var output = Think(trainingInputs);
            var error = PerformOperation(trainingOutputs, output, OPERATION.Subtract);
            var adjustment = DotProduct(Transpose(trainingInputs),
                PerformOperation(error, Activate(output, true), OPERATION.Multiply));
            _weights = PerformOperation(_weights, adjustment, OPERATION.Add);
        }
    }

    private double[,] DotProduct(double[,] matrix1, double[,] matrix2)
    {
        var numberOfRowsInMatrix1 = matrix1.GetLength(0);
        var numberOfColumnsInMatrix1 = matrix1.GetLength(1);
        var numberOfRowsInMatrix2 = matrix2.GetLength(0);
        var numberOfColumnsInMatrix2 = matrix2.GetLength(1);
        var result = new double[numberOfRowsInMatrix1, numberOfColumnsInMatrix2];

        for (var rowInMatrix1 = 0; rowInMatrix1 < numberOfRowsInMatrix1; rowInMatrix1++)
        {
            for (var columnInMatrix2 = 0; columnInMatrix2 < numberOfColumnsInMatrix2; columnInMatrix2++)
            {
                double sum = 0;
                for (var columnInMatrix1 = 0; columnInMatrix1 < numberOfColumnsInMatrix1; columnInMatrix1++)
                {
                    sum += matrix1[rowInMatrix1, columnInMatrix1] * matrix2[columnInMatrix1, columnInMatrix2];
                }

                result[rowInMatrix1, columnInMatrix2] = sum;
            }
        }

        return result;
    }

    public double[,] Think(double[,] inputs)
    {
        return Activate(DotProduct(inputs, _weights), false);
    }

    private double[,] PerformOperation(double[,] matrix1, double[,] matrix2, OPERATION operation)
    {
        var rows = matrix1.GetLength(0);
        var columns = matrix1.GetLength(1);

        var result = new double[rows, columns];

        for (var i = 0; i < rows; i++)
        {
            for (var j = 0; j < columns; j++)
            {
                switch (operation)
                {
                    case OPERATION.Multiply:
                        result[i, j] = matrix1[i, j] * matrix2[i, j];
                        break;
                    case OPERATION.Add:
                        result[i, j] = matrix1[i, j] + matrix2[i, j];
                        break;
                    case OPERATION.Subtract:
                        result[i, j] = matrix1[i, j] - matrix2[i, j];
                        break;
                    default:
                        throw new ArgumentOutOfRangeException(nameof(operation), operation, null);
                }
            }
        }
        return result;
    }

    private double[,] Transpose(double[,] matrix)
    {
        return matrix.Cast<double>().ToArray().Transpose(matrix.GetLength(0), matrix.GetLength(1));
    }

    static void Main(string[] args)
    {
        var neuralNetwork = new NeuralNetwork();
        var trainingSetInputs = new double[,] { { 0, 0, 0 }, { 1, 1, 1 }, {1,0,0} };
        var trainingSetOutputs = new double[,] { { 0 }, { 1 }, { 1 } };
        neuralNetwork.Train(trainingSetInputs, trainingSetOutputs, 1000);
        var output = neuralNetwork.Think(new double[,] { { 0, 1, 0 }, { 0, 0, 0 }, { 0, 0, 1 } });
        PrintMatrix(output);
    }

    static void PrintMatrix(double[,] matrix)
    {
        var rows = matrix.GetLength(0);
        var columns = matrix.GetLength(1);
        for (var row = 0; row < rows; row++)
        {
            for (var column = 0; column < columns; column++)
            {
                Console.Write(Math.Round(matrix[row,column]) + " ");
            }
            Console.WriteLine();
        }
    }
}