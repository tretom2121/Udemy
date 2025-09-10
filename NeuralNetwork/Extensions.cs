namespace NeuralNetwork;

public static class Extensions
{
    public static double[,] Transpose(this double[] array, int rows, int columns)
    {
        var result = new double[columns, rows];
        for (var row = 0; row < rows; row++)
        {
            for (var column = 0; column < columns; column++)
            {
                result[column, row] = array[row * column + column];
            }
        }
        return result;
    }
}