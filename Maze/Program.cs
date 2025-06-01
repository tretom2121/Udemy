using System.Runtime.InteropServices;
using TorchSharp;
 
int[,] maze1 =
{
    //0   1   2   3   4   5   6   7   8   9   10  11
    { 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0 }, //row 0
    { 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0 }, //row 1
    { 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0 }, //row 2
    { 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0 }, //row 3
    { 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0 }, //row 4
    { 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0 }, //row 5
    { 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0 }, //row 6
    { 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0 }, //row 7
    { 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0 }, //row 8
    { 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0 }, //row 9
    { 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0 }, //row 10
    { 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 } //row 11 (start position is (11, 5))
};
 
const string UP = "up";
const string DOWN = "down";
const string LEFT = "left";
const string RIGHT = "right";
 
string[] actions = [UP, DOWN, LEFT, RIGHT];
int[,] rewards;
 
const int WALL_REWARD_VALUE = -500;
const int FLOOR_REWARD_VALUE = -10;
const int GOAL_REWARD_VALUE = 500;
 
void SetupRewards(int[,] maze, int wallValue, int floorValue, int goalValue)
{
    var mazeRows = maze.GetLength(0);
    var mazeColumns = maze.GetLength(1);
 
    rewards = new int[mazeRows, mazeColumns];
 
    for (var i = 0; i < mazeRows; i++)
    {
        for (var j = 0; j < mazeColumns; j++)
        {
            switch (maze[i, j])
            {
                case 0:
                    rewards[i, j] = wallValue;
                    break;
                case 1:
                    rewards[i, j] = floorValue;
                    break;
                case 2:
                    rewards[i, j] = goalValue;
                    break;
            }
        }
    }
}
 
torch.Tensor qValues;
 
void SetupQValues(int[,] maze)
{
    var mazeRows = maze.GetLength(0);
    var mazeColumns = maze.GetLength(1);
 
    qValues = torch.zeros(mazeRows, mazeColumns, 4);
}
 
bool HasHitWallOrEndOfMaze(int currentRow, int currentColumn, int floorValue)
{
    return rewards[currentRow, currentColumn] != floorValue;
}
 
long DetermineNextAction(int currentRow, int currentColumn, float epsilon)
{
    var random = new Random();
    var randomBetween0and1 = random.NextDouble();
 
    var returnVal =  randomBetween0and1 < epsilon ? torch.argmax(qValues[currentRow, currentColumn]).item<long>() : random.Next(4);
    return returnVal;
}
 
(int, int) MoveOneSpace(int[,] maze, int currentRow, int currentColumn, long currentAction)
{
    var mazeRows = maze.GetLength(0);
    var mazeColumns = maze.GetLength(1);
    var nextRow = currentRow;
    var nextColumn = currentColumn;
 
    if (actions[currentAction] == UP && currentRow > 0)
    {
        nextRow--;
    }
    else if (actions[currentAction] == DOWN && currentRow < mazeRows - 1)
    {
        nextRow++;
    }
    else if (actions[currentAction] == LEFT && currentColumn > 0)
    {
        nextColumn--;
    }
    else if (actions[currentAction] == RIGHT && currentColumn < mazeColumns - 1)
    {
        nextColumn++;
    }
 
    return (nextRow, nextColumn);
}
 
void TrainTheModel(int[,] maze, int floorValue, float epsilon, float discountFactor, float learningRate, float episodes)
{
    for (var episode = 0; episode < episodes; episode++)
    {
        Console.WriteLine($"-----------Starting episode: {episode} -----------");
 
        var currentRow = 11;
        var currentColumn = 5;
        while (!HasHitWallOrEndOfMaze(currentRow, currentColumn, floorValue))
        {
            var currentAction = DetermineNextAction(currentRow, currentColumn, epsilon);
            var previousRow = currentRow;
            var previousColumn = currentColumn;
            var nextMove = MoveOneSpace(maze, currentRow, currentColumn, currentAction);
            currentRow = nextMove.Item1;
            currentColumn = nextMove.Item2;
            float reward = rewards[currentRow, currentColumn];
            var previousQValue = qValues[previousRow, previousColumn, currentAction].item<float>();
            var temporalDifference = reward + (discountFactor * torch.max(qValues[currentRow, currentColumn])).item<float>() - previousQValue;
            var nextQValue = previousQValue + learningRate * temporalDifference;
            qValues[previousRow, previousColumn, currentAction] = nextQValue;
        }
 
        Console.WriteLine($"-----------Finished episode: {episode} -----------");
    }
 
    Console.WriteLine("Training completed");
}
 
List<int[]> NavigateMaze(int[,] maze, int startRow, int startColumn, int floorValue, int wallValue)
{
    var path = new List<int[]>();
 
    if (HasHitWallOrEndOfMaze(startRow, startColumn, floorValue))
    {
        return [];
    }
    else
    {
        var currentRow = startRow;
        var currentColumn = startColumn;
        path = [[currentRow, currentColumn]];
 
        while (!HasHitWallOrEndOfMaze(currentRow, currentColumn, floorValue))
        {
            var nextAction = (int)DetermineNextAction(currentRow, currentColumn, 1.0f);
            var nextMove = MoveOneSpace(maze, currentRow, currentColumn, nextAction);
            currentRow = nextMove.Item1;
            currentColumn = nextMove.Item2;
            if (rewards[currentRow, currentColumn] != wallValue)
            {
                path.Add([currentRow, currentColumn]);
            }
            else
            {
                continue;
            }
        }
    }
 
    var moveCount = 1;
    for (var i = 0; i < path.Count; i++)
    {
        Console.Write($"Move {moveCount}: (");
        foreach (var element in path[i])
        {
            Console.Write($"{element}");
        }
        Console.Write($")");
        Console.WriteLine();
        moveCount++;
    }
 
    return path;
}
 
const float EPSILON = 0.95f;
const float DISCOUNT_FACTOR = 0.8f;
const float LEARNING_RATE = 0.9f;
const  int EPISODES = 1500;
const int START_ROW = 11;
const int START_COLUMN = 5;
 
SetupRewards(maze1, WALL_REWARD_VALUE, FLOOR_REWARD_VALUE, GOAL_REWARD_VALUE);
SetupQValues(maze1);
TrainTheModel(maze1, FLOOR_REWARD_VALUE, EPSILON, DISCOUNT_FACTOR, LEARNING_RATE, EPISODES);
NavigateMaze(maze1, START_ROW, START_COLUMN, FLOOR_REWARD_VALUE, WALL_REWARD_VALUE);