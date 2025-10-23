using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Vision;
using static Microsoft.ML.DataOperationsCatalog;

public class ImageData
{
    [LoadColumn(0)]
    public string? ImagePath { get; set; }

    [LoadColumn(1)]
    public string? Label { get; set; }
}

public class InputData
{
    public byte[] Image { get; set; }
    public uint LabelKey { get; set; }
    public string ImagePath { get; set; }
    public string Label { get; set; }
}

class Output
{
    public string ImagePath { get; set; }
    public string Label { get; set; }
    public string PredictedLabel { get; set; }
}

public class Program
{
    private static string dataFolder = "D:\\Code\\Udemy\\ImageClassification\\Data";

    private static IEnumerable<ImageData> LoadImagesFromDirectory(string folder)
    {
        var files = Directory.GetFiles(folder, "*", searchOption: SearchOption.AllDirectories);
        foreach (var file in files)
        {
            if (Path.GetExtension(file) != ".jpg" && Path.GetExtension(file) != ".jpeg" && Path.GetExtension(file) != ".png") { continue; }

            string label = Path.GetFileNameWithoutExtension((file).Trim());
            label = label.Substring(0, label.Length - 1);

            yield return new ImageData
            {
                ImagePath = file,
                Label = label
            };
        }
    }

    public static void PrintDataView(IDataView dataView)
    {
        var preview = dataView.Preview();
        foreach (var row in preview.RowView)
        {
            foreach (var kvp in row.Values)
            {
                Console.WriteLine($"{kvp.Key}: {kvp.Value}");
            }

            Console.WriteLine();
        }
    }

    private static void OutputPrediction(Output prediction)
    {
        var imageName = Path.GetFileName(prediction.ImagePath);
        Console.WriteLine($"Image: {imageName} | Actual label: {prediction.Label} | Predicted label: {prediction.PredictedLabel}");
    }

    private static void ClassifyMultiple(MLContext mlContext, IDataView data, ITransformer trainedModel)
    {
        var predictionData = trainedModel.Transform(data);
        var predictions = mlContext.Data.CreateEnumerable<Output>(predictionData, reuseRowObject: false).ToList();

        Console.WriteLine("AI predictions:");
        foreach (var prediction in predictions.Take(4))
        {
            OutputPrediction(prediction);
        }
    }
    
    public static void Main()
    {
        var mlContext = new MLContext();
        var images = LoadImagesFromDirectory(folder: dataFolder);
        var imageData = mlContext.Data.LoadFromEnumerable(images);
        var shuffleData = mlContext.Data.ShuffleRows(imageData);

        var preprocessingPipeline = mlContext.Transforms.Conversion
            .MapValueToKey(inputColumnName: "Label", outputColumnName: "LabelKey")
            .Append(mlContext.Transforms.LoadRawImageBytes("Image", dataFolder, "ImagePath"));
        
        var preprocessedData = preprocessingPipeline.Fit(shuffleData).Transform(shuffleData);
        var trainTestSplit = mlContext.Data.TrainTestSplit(preprocessedData, testFraction: 0.4);
        var trainSet = trainTestSplit.TrainSet;
        var testSet = trainTestSplit.TestSet;

        var classifierOptions = new ImageClassificationTrainer.Options()
        {
            FeatureColumnName = "Image",
            LabelColumnName = "LabelKey",
            Arch = ImageClassificationTrainer.Architecture.ResnetV2101,
            MetricsCallback = Console.WriteLine,
            TestOnTrainSet = false,
            ValidationSet = testSet,
            ReuseTrainSetBottleneckCachedValues = true,
            ReuseValidationSetBottleneckCachedValues = true
        };

        var trainPipeline = mlContext.MulticlassClassification.Trainers.ImageClassification(classifierOptions)
            .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
        
        var trainedModel = trainPipeline.Fit(trainSet);
        
        ClassifyMultiple(mlContext, testSet, trainedModel);
    }
}