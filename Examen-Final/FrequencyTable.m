function Table = FrequencyTable(xi,N,bands)
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here
    xi = sort(xi);
    Min = xi(1); Max = xi(end);
    range = Max - Min;
    bandwidth = range/bands;
    AbsoluteFrequencies = zeros(1,bands);
    lo = Min;
    up = Min + bandwidth;
    intervals = zeros(bands,2);
    for j =1:bands
        if lo == Min
            ind = xi(xi >= lo);
        else
            ind = xi(xi > lo);
        end
        ind = ind(ind<=up);
        AbsoluteFrequencies(j) = length(ind);
        intervals(j,:) = [lo up];
        lo = up; up = up + bandwidth;
        
    end
    RelativeFrequencies = zeros(1,bands);
    AcumRelativeFrequencies = zeros(1,bands);
    acum = 0;
    for j =1:bands
        RelativeFrequencies (j) = AbsoluteFrequencies(j)/N;
        acum = acum + RelativeFrequencies (j);
        AcumRelativeFrequencies(j) = acum;
    end
    

    %Output storage
    Table.Intervals = intervals;
    Table.AbsoluteFrequencies = AbsoluteFrequencies;
    Table.RelativeFrequencies = RelativeFrequencies;
    Table.AcumRelativeFrequencies = AcumRelativeFrequencies;

end