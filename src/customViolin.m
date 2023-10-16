function customViolin(data,violinWidth,groupLabels,plotTitle)

%Setting Parameters
groupSpacing=0.1;
color_map=colormap(gray(5));

[~,numGroups]=size(data);
violinPlots=cell(1,numGroups);
quartile_width=0.01; %Width of the quartile box
median_height=1.5; %Height of the box of the median line

figure;

    %Loop through each column of data and create the violin plots
    for i=[1:numGroups]
        [f,xi]=ksdensity(data(:,i));
    
        %Create the violin outline
        yOutline = [xi, fliplr(xi)];
        xOutline = [-f, fliplr(f)];
    
        %Offset the violins for different groups
        xViolin = xOutline + (i - 1) * groupSpacing;
        % Plot the violin outline
        violinPlots{i} = fill(xViolin, yOutline, 'b', 'EdgeColor', 'k', 'FaceAlpha', 0.5,'FaceColor',color_map(i+1,:));

        
        %Calculate the quartiles, median, maximum, and minimum
        quartiles = quantile(data(:,i), [0.25, 0.75]);
        medianValue = median(data(:,i),'omitnan');
        maxValue = max(data(:,i));
        minValue = min(data(:,i));

        %Getting Data for plotting the median boxpplot
        %violin_width=max(xViolin)-min(xViolin);
        %quartile_width=violin_width*0.04;

        % Plot the boxplot elements as rectangles and circles

        %Plotting the interquartile box
        rectangle('position',[(i-1)*groupSpacing-quartile_width/2,quartiles(1),quartile_width,quartiles(2)-quartiles(1)],'FaceColor','k');
    
        %Plotting the wiskers
        line([(i-1)*groupSpacing,(i-1)*groupSpacing],[minValue,maxValue],'Color', 'k', 'LineWidth', 1)
        
        %Plotting the median

        rectangle('Position',[(i-1)*groupSpacing-quartile_width/2,medianValue-quartile_width/2,quartile_width,median_height],'FaceColor','w','EdgeColor','none')


      

        hold on
    
    
    end

    % Set the x-axis labels and title
    xticks(groupSpacing * (0:numGroups - 1));
    xticklabels(groupLabels);
    xlabel('Groups');
    title(plotTitle);

end