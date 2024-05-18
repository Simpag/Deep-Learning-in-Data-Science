function [data, unique_data, char_to_ind, ind_to_char] = ReadData(location)
    fid = fopen(location, 'r');
    data = fscanf(fid,'%c');
    fclose(fid);

    unique_data = unique(data);

    char_to_ind = containers.Map('KeyType','char','ValueType','int32');
    ind_to_char = containers.Map('KeyType','int32','ValueType','char');

    for i = 1:length(unique_data)
        char_to_ind(unique_data(i)) = i;
        ind_to_char(i) = unique_data(i);
    end
end