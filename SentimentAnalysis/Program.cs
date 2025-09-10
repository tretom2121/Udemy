using Microsoft.ML;
using Microsoft.ML.Data;
using System.IO;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Trainers;

namespace SentimentAnalysis
{
    public class Program
    {
        public static void Main(string[] args)
        {
            var mlContext = new MLContext();
            
            string dataPath = "train.csv";
            // PrepareTrainingData(dataPath);
            // IDataView dataView = mlContext.Data.LoadFromTextFile<MovieReview>(dataPath, hasHeader: true, allowQuoting: true, separatorChar: ',');
            // var model = TrainModel(mlContext, dataView);
            // mlContext.Model.Save(model, dataView.Schema, "sentiment_model.zip");

            string modelPath = "sentiment_model.zip";
            string testDataPath = "test.csv";
            ITransformer model;
            using (var stream = new FileStream(modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                model = mlContext.Model.Load(stream, out var modelInputSchema);
            }

            IDataView testData = mlContext.Data.LoadFromTextFile<TextData>(testDataPath, separatorChar: ',', hasHeader: true);
            var predictor = mlContext.Model.CreatePredictionEngine<TextData, SentimentPrediction>(model);
            
            var testDataList = mlContext.Data.CreateEnumerable<TextData>(testData, reuseRowObject: false).ToList();
            foreach (var data in testDataList)
            {
                var prediction = predictor.Predict(data);
                Console.WriteLine($"Text: {data.Text} | Sentiment: {(prediction.IsPositiveSentiment ? "Positive" : "Negative")}");
            }
        }

        private static TransformerChain<BinaryPredictionTransformer<CalibratedModelParametersBase<LinearBinaryModelParameters, PlattCalibrator>>> TrainModel(MLContext mlContext, IDataView dataView)
        {
            var pipeline = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: "Text").Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression("Label", "Features"));
            var model = pipeline.Fit(dataView);
            var predictions = model.Transform(dataView);
            var metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"Precision: {metrics.PositivePrecision:P2}");
            Console.WriteLine($"Recall: {metrics.PositiveRecall}");
            Console.WriteLine($"F1-score: {metrics.F1Score:P2}");
            return model;
        }

        private static void PrepareTrainingData(string dataPath)
        {
            string text = File.ReadAllText(dataPath);
            
            using (StreamReader sr = new StreamReader(dataPath))
            {
                text = text.Replace("\'", "");
            }
            
            File.WriteAllText((dataPath), text);
        }

        public class TextData
        {
            [LoadColumn(0)]
            public string Text { get; set; }
        }

        public class SentimentPrediction
        {
            [ColumnName("Score")]
            public float SentimentScore { get; set; }

            public bool IsPositiveSentiment => SentimentScore < 0.5f;
        }
    }
}