%----------Supp Fig 1--------

plot(X_joint,Y_joint, '--', 'Color', [1 0.5 0], 'LineWidth', 1); %ROC joint
hold on 
plot(X_noninv,Y_noninv, '-', 'Color', 'g', 'LineWidth', 1); % ROC non invasive
plot(X_inv,Y_inv, '-', 'Color', [0.6 0.6 0.6], 'LineWidth', 1); % ROC invasive

plot(OptPoint_joint(1), OptPoint_joint(2), 'd', 'MarkerEdgeColor', 'k',...
    'MarkerFaceColor', [1 0.5 0],'MarkerSize', 8) %optimal joint
plot(OptPoint_noninv(1), OptPoint_noninv(2), 'd','MarkerEdgeColor', 'k', ...
    'MarkerFaceColor', 'g', 'MarkerSize', 8) % optimal non invasive 
plot(train_OptPoint_inv(1), train_OptPoint_inv(2), 'd','MarkerEdgeColor', 'k',...
    'MarkerSize', 8, 'MarkerFaceColor', [0.6 0.6 0.6]) % Optimal invasive

plot([0 1], [0 1], '--', 'color', [0.7 0.7 0.7]) % Chance line

hold off
legend({'Joint model', 'Non-invasive model', 'Invasive model',...
    'Joint model optimal operation', 'Non-invasive optimal operation', 'Invasive optimal operation'}, ...
    'Location', 'southeast', 'Box', 'off');
grid on;
ax = gca;
ax.XTick = 0:0.2:1;
ax.Box = 'off';
save2pdf('ROC curves ensemble_newdat')


%---------pred importance---------


figure
hold on

plot(mdl.predictorImportance, '--s', 'Markersize', 10, 'MarkerEdgeColor', 'b', 'MarkerFaceColor', 'g',...
    'Color', 'k')
xticks(1:width(train_dat));
xticklabels(var_names)
xtickangle(45)
ax = gca;
ax.Box = 'off';
ax.XLim = [1 length(var_names)]; 
xli = ax.XLim;
yli = ax.YLim;
patch([xli(1)-0.5 11.5 11.5 xli(1)-0.5], [yli(1) yli(1) yli(2) yli(2)], 'g', 'FaceAlpha', 0.2, ...
    'EdgeColor', 'none')
patch([11.5 xli(2)+0.5 xli(2)+0.5 11.5], [yli(1) yli(1) yli(2) yli(2)], 'w','FaceColor', [1 0.5 0], 'FaceAlpha', 0.2, ...
    'EdgeColor', 'none')
xlim([0.5 37.5])
save2pdf('FeatureWeightsEnsemble')