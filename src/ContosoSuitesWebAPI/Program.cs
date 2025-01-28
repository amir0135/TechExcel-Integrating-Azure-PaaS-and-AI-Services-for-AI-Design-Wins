using Azure.Identity;
using Microsoft.Azure.Cosmos;
using ContosoSuitesWebAPI.Agents;
using ContosoSuitesWebAPI.Entities;
using ContosoSuitesWebAPI.Plugins;
using ContosoSuitesWebAPI.Services;
using Microsoft.Data.SqlClient;
using Azure.AI.OpenAI;
using Azure;
using Microsoft.AspNetCore.Mvc;

// NEW Semantic Kernel namespaces
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Connectors.OpenAI;
using Microsoft.SemanticKernel.ChatCompletion;

var builder = WebApplication.CreateBuilder(args);

// 1) Build configuration from user secrets + environment vars
var config = new ConfigurationBuilder()
    .AddUserSecrets<Program>()     
    .AddEnvironmentVariables()
    .Build();

// Add services to the container.
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

// Use dependency injection to inject services into the application.
builder.Services.AddSingleton<IVectorizationService, VectorizationService>();
builder.Services.AddSingleton<MaintenanceCopilot, MaintenanceCopilot>();

// Create a single instance of the DatabaseService to be shared across the application.
builder.Services.AddSingleton<IDatabaseService, DatabaseService>((sp) =>
{
    // read the connection string from config/builder.Configuration
    var connectionString = builder.Configuration.GetConnectionString("ContosoSuites");
    return new DatabaseService(connectionString!);
});

// Create a single instance of the CosmosClient
builder.Services.AddSingleton<CosmosClient>((sp) =>
{
    string userAssignedClientId = builder.Configuration["AZURE_CLIENT_ID"]!;
    var credential = new DefaultAzureCredential(
        new DefaultAzureCredentialOptions
        {
            ManagedIdentityClientId = userAssignedClientId
        });
    CosmosClient client = new(
        accountEndpoint: builder.Configuration["CosmosDB:AccountEndpoint"]!,
        tokenCredential: credential
    );
    return client;
});

// Create a single instance of the AzureOpenAIClient
builder.Services.AddSingleton<AzureOpenAIClient>((sp) =>
{
    var endpoint = new Uri(builder.Configuration["AzureOpenAI:Endpoint"]!);
    var credentials = new AzureKeyCredential(builder.Configuration["AzureOpenAI:ApiKey"]!);
    return new AzureOpenAIClient(endpoint, credentials);
});

// 2) Create the Semantic Kernel singleton
builder.Services.AddSingleton<Kernel>((sp) =>
{
    IKernelBuilder kernelBuilder = Kernel.CreateBuilder();

    // read AzureOpenAI config from environment or user-secrets
    var deploymentName = builder.Configuration["AzureOpenAI:DeploymentName"]!;
    var endpoint = builder.Configuration["AzureOpenAI:Endpoint"]!;
    var apiKey = builder.Configuration["AzureOpenAI:ApiKey"]!;

    // Add Azure OpenAI chat completion
    kernelBuilder.AddAzureOpenAIChatCompletion(
        deploymentName: deploymentName,
        endpoint: endpoint,
        apiKey: apiKey
    );

    // (NEW) Add Azure OpenAI text embedding generation 
    // feature is experimental, so we suppress warnings:
#pragma warning disable SK_FEATURE_EXPERIMENTAL
    kernelBuilder.AddAzureOpenAITextEmbeddingGeneration(
        deploymentName: builder.Configuration["AzureOpenAI:EmbeddingDeploymentName"]!,
        endpoint: builder.Configuration["AzureOpenAI:Endpoint"]!,
        apiKey: builder.Configuration["AzureOpenAI:ApiKey"]!
    );
#pragma warning restore SK_FEATURE_EXPERIMENTAL

    // Add the DatabaseService as a plugin (calls [KernelFunction] methods)
    var databaseService = sp.GetRequiredService<IDatabaseService>();
    kernelBuilder.Plugins.AddFromObject(databaseService);

    return kernelBuilder.Build();
});

var app = builder.Build();

// Configure the HTTP request pipeline
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.UseHttpsRedirection();

/**** Endpoints ****/
// Default landing page for the API
app.MapGet("/", async () =>
{
    return "Welcome to the Contoso Suites Web API!";
})
    .WithName("Index")
    .WithOpenApi();

/****** HOTELS ENDPOINTS ******/

// Retrieve the set of hotels from the database
app.MapGet("/Hotels", async () =>
{
    var hotels = await app.Services.GetRequiredService<IDatabaseService>().GetHotels();
    return hotels;
})
    .WithName("GetHotels")
    .WithOpenApi();

// Retrieve the bookings for a specific hotel
app.MapGet("/Hotels/{hotelId}/Bookings/", async (int hotelId) =>
{
    var bookings = await app.Services.GetRequiredService<IDatabaseService>().GetBookingsForHotel(hotelId);
    return bookings;
})
    .WithName("GetBookingsForHotel")
    .WithOpenApi();

// Retrieve the bookings for a specific hotel that are after a specified date
app.MapGet("/Hotels/{hotelId}/Bookings/{min_date}", async (int hotelId, DateTime min_date) =>
{
    var bookings = await app.Services.GetRequiredService<IDatabaseService>().GetBookingsByHotelAndMinimumDate(hotelId, min_date);
    return bookings;
})
    .WithName("GetRecentBookingsForHotel")
    .WithOpenApi();

/****** OTHER ENDPOINTS ******/

// (3) The /Chat POST endpoint using Semantic Kernel
app.MapPost("/Chat", async Task<string> (HttpRequest request) =>
{
    // read the user's prompt from form data
    var message = await Task.FromResult(request.Form["message"]);

    // get the Semantic Kernel
    var kernel = app.Services.GetRequiredService<Kernel>();

    // get the chat completion service
    var chatCompletionService = kernel.GetRequiredService<IChatCompletionService>();

    // auto-invoke any [KernelFunction] method that matches the user request
    var executionSettings = new OpenAIPromptExecutionSettings
    {
        ToolCallBehavior = ToolCallBehavior.AutoInvokeKernelFunctions
    };

    // ask the chat completion service to respond
    var response = await chatCompletionService.GetChatMessageContentAsync(
        message.ToString(),
        executionSettings,
        kernel
    );

    // return the final content
    return response?.Content!;
})
    .WithName("Chat")
    .WithOpenApi();

// This endpoint is used to vectorize a text string
app.MapGet("/Vectorize", async (string text, [FromServices] IVectorizationService vectorizationService) =>
{
    var embeddings = await vectorizationService.GetEmbeddings(text);
    return embeddings;
})
    .WithName("Vectorize")
    .WithOpenApi();

// This endpoint is used to search for maintenance requests based on a vectorized query
app.MapPost("/VectorSearch", async ([FromBody] float[] queryVector, [FromServices] IVectorizationService vectorizationService, int max_results = 0, double minimum_similarity_score = 0.8) =>
{
    // Exercise 3 Task 3 TODO #3: Call the ExecuteVectorSearch function
    var results = await vectorizationService.ExecuteVectorSearch(queryVector, max_results, minimum_similarity_score);
    return results;
})
    .WithName("VectorSearch")
    .WithOpenApi();

// This endpoint is used to send a message to the Maintenance Copilot
app.MapPost("/MaintenanceCopilotChat", async ([FromBody]string message, [FromServices] MaintenanceCopilot copilot) =>
{
    // not yet implemented
    throw new NotImplementedException();
})
    .WithName("Copilot")
    .WithOpenApi();

app.Run();