function customViolin(data,groupLabels,plotTitle)

%Setting Parameters
groupSpacing=0.9;
right_left_spacing=1.1;
color_map=colormap(winter(7));

[~,numGroups]=size(data);
middle_group=floor(numGroups/2);
violinPlots=cell(1,numGroups);
quartile_width=0.15; %Width of the quartile box
median_height=0.2; %Height of the box of the median line
plot_middle=0;
tick_numbers=[];
color_ind=1;

figure;

    %Loop through each column of data and create the violin plots
    for i=[1:numGroups]

        spacing=groupSpacing;

        if i==1+middle_group
            color_ind=1;
            spacing=right_left_spacing;

        end
        plot_middle=plot_middle+spacing;
        tick_numbers=[tick_numbers,plot_middle];
        [f,xi]=ksdensity(data(:,i));
    
        %Create the violin outline
        yOutline = [xi, fliplr(xi)];
        xOutline = [-f, fliplr(f)];
    
        %Offset the violins for different groups
        xViolin = xOutline + plot_middle;
        % Plot the violin outline
        violinPlots{i} = fill(xViolin, yOutline, 'b', 'EdgeColor', 'k', 'FaceAlpha', 0.4,'FaceColor',color_map(color_ind+3,:),'LineWidth',1.5);

        
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
        rectangle('position',[plot_middle-quartile_width/2,quartiles(1),quartile_width,quartiles(2)-quartiles(1)],'FaceColor','k','EdgeColor','none');
    
        %Plotting the wiskers
        line([plot_middle,plot_middle],[minValue,maxValue],'Color', 'k', 'LineWidth', 1)
        
        %Plotting the median

        rectangle('Position',[plot_middle-quartile_width/2,medianValue-quartile_width/2,quartile_width,median_height],'FaceColor','w','EdgeColor','none')



        color_ind=color_ind+1;
        hold on
    
    
    end

    % Set the x-axis labels and title

    xticks(tick_numbers);
    xticklabels(groupLabels);
    xtickangle(15);

    % x-labels
    left_ind=mean(tick_numbers(1:middle_group));
    t=text(left_ind,-3.5,'Left Eye','HorizontalAlignment','center','FontName','Helvetica','FontSize',14,'FontWeight','bold');

    right_ind=mean(tick_numbers(middle_group+1:end));
    t=text(right_ind,-3.5,'Right Eye','HorizontalAlignment','center','FontName','Helvetica','FontSize',14,'FontWeight','bold');

    title(plotTitle,'FontName','Helvetica','FontSize',16);
    a = get(gca,'XTickLabel');
    set(gca,'XTickLabel',a,'FontName','Helvetica','fontsize',12)

    %y-label
    ylabel('DVA Error (degrees)','FontName','Helvetica','FontSize',14,'FontWeight','bold')

end