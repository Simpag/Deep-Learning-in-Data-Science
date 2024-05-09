function PlotResults(NetParams, costs_train, costs_val, loss_train, loss_val, acc_train, acc_val, etas, time, test_accuracy, val_accuracy)
    if (length(costs_train) < 1)
        disp("Logging must be on to plot!")
        return;
    end

    % Plotting
    scr_siz = get(0,'ScreenSize');
    f = figure;
    f.Position = floor([150 150 scr_siz(3)*0.8 scr_siz(4)*0.8]);
    T = tiledlayout(f, 2, 2);
    title(T, "Test accuracy: " + test_accuracy + ", Validation accuracy: " + val_accuracy);

    % Plot train-validation losses
    nexttile(T);
    plot(time, loss_train, time, loss_val);
    legend("Training loss", "Validation loss");
    %ylim([min(loss_train) * 0.9,max(loss_train) * 1.1]);
    grid();
    xlabel("Update step")
    ylabel("Loss")
    fontsize(T,24,"points")

    % Plot train-validation costs
    nexttile(T);
    plot(time, costs_train, time, costs_val);
    legend("Training cost", "Validation cost");
    %ylim([min(costs_train) * 0.9,max(costs_train) * 1.1]);
    grid();
    xlabel("Update step")
    ylabel("Cost")
    fontsize(T,24,"points")

    % Plot train-validation accuracy
    nexttile(T);
    plot(time, acc_train, time, acc_val);
    legend("Training accuracy", "Validation accuracy");
    grid();
    xlabel("Update step")
    ylabel("Accuracy")
    fontsize(T,24,"points")

    % Plot learning rate
    nexttile(T);
    plot(time, etas);
    grid();
    xlabel("Update step")
    ylabel("\eta")
    fontsize(T,24,"points")
end