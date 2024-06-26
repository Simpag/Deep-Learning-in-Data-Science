function search_plotter()
    % string(val_accuracy) + ";" + string(NP.lambda) + ";" + string( NP.shiftProb) + ";" + string(NP.mirrorProb) + ";" + string(NP.hidden_nodes) + ";" + string(NP.dropout) + ";" + string(NP.eta)

    res = readmatrix("C:\Users\Simon\Documents\Github\Deep-Learning-in-Data-Science\Assignment2/random_search_bonus_adam_2.txt");
    acc = res(:, 1);
    lambda = res(:,2);
    shift = res(:,3);
    mirror = res(:,4);
    hidden = res(:,5);
    drop = res(:,6);
    eta = res(:,7);
    
    subplot(2,3,1);
    title("Lambda")
    plot(log10(lambda), acc, '*')
    xlabel("log(\lambda)")
    ylabel("Accuracy")

    subplot(2,3,2)
    title("Shift")
    plot(shift, acc, '*')
    xlabel("Shift Probability")
    ylabel("Accuracy")

    subplot(2,3,3)
    title("mirror")
    plot(mirror, acc, '*')
    xlabel("Mirror Probability")
    ylabel("Accuracy")

    subplot(2,3,4)
    title("hidden")
    plot(hidden, acc, '*')
    xlabel("Hidden Nodes")
    ylabel("Accuracy")

    subplot(2,3,5)
    title("drop")
    plot(drop, acc, '*')
    xlabel("Droprate")
    ylabel("Accuracy")
    
    subplot(2,3,6)
    title("eta")
    plot(eta, acc, '*')
    xlabel("\eta")
    ylabel("Accuracy")
end