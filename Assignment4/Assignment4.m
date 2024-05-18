function assignment4()
    rng(1)
    % Load the data
    [data, unique_data, char_to_ind, ind_to_char] = ReadData("data\goblet_book.txt");

    NetParams = struct();

    NetParams.K = length(unique_data);  % Output dimensionality
    NetParams.m = 100;                  % hidden state dimensionality
    NetParams.eta = 0.1;                % learning rate
    NetParams.seq_length = 25;          % training sequence length
    NetParams.epochs = 2;


    RNN = InitializeNetwork(NetParams, 0.01);

    [RNN, losses] = Train(RNN, NetParams, data, ind_to_char, char_to_ind);
    PlotResults(losses);
    file_name = strrep("RNN_model_" + string(datetime("now")) + ".xml", " ", "_");
    file_name = strrep(file_name, ":", "-");
    save(file_name, '-struct', 'RNN');

    h_0 = zeros(NetParams.m, 1);
    x_0 = OneHot(char_to_ind("."), NetParams.K)
    disp("Final text generation: ")
    disp(SynthesizeText(RNN, h_0, x_0, 1000, ind_to_char))
end

function PlotResults(losses)
    % Plotting
    plot(1:length(losses), losses);
    title("Training Loss");
    grid();
end