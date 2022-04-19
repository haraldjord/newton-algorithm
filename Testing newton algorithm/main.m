%% Newton algorithm for solving rosenbrock function
% parameters
maxiter = 10^4; %'largest allowed number of iterations k -- some large number';
grad_tol = 10^(-4); %'how small we want ||nabla f|| to be (close to the solution) -- some small number';
param = [maxiter, grad_tol];

%first startingpoint
x0 = [1.2, 1.2]';
[x_opt, fval_opt, x_iter, f_iter, alpha] = newton_algorithm(x0,param);

figure(1)
plot_iter_rosenbrock(x_iter,str)
disp('newton algorithm uses')
disp(size(x_iter,2))
disp('iterations')

%second starting point
x0 = [-1.1, 1]';
[x_opt, fval_opt, x_iter, f_iter, alpha] = newton_algorithm(x0,param);

figure(2)
plot_iter_rosenbrock(x_iter)
disp('newton algorithm uses')
disp(size(x_iter,2))
disp('iterations')
