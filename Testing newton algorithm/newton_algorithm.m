function [x_opt, fval_opt, x_iter, f_iter, alpha] = newton_algorithm(x0,param)
% Returns 
% x_opt: x^*
% fval_opt: f(x^*)
% x_iter: all iterates k -- column k contains x_k
% f_iter: a vector of all function values f(x_k)
% alpha: a vector of all step lenghts alpha_k

% Termination criteria
maxiter = param(1); 
grad_tol = param(2); 

x0 = x0(:);
n = size(x0,1); % Number of variables

% Declare some variables
x =     NaN(n,maxiter);
p =     NaN(n,maxiter);
grad =  NaN(n,maxiter);
alpha = NaN(1,maxiter);
fval =  NaN(1,maxiter);

k = 1; % iteration number
x(:,k) = x0; %'store the initial point.';
grad(:,k) = gradient(x(:,k)); %'store the gradient';

while (k < maxiter) && (norm(grad(:,k)) > grad_tol) %'maxiter not exceeded' && 'the norm of the gradient larger than grad_tol'
    fval(k) = f(x(:,k));  % Evaluate the Rosenbrock function
    grad(:,k) = gradient(x(:,k));
    hes = hessian(x(:,k));
    p(:,k) = - hes\grad(:,k);
    alpha_0 =  1;  %initial guess of step lenght 
    alpha(k) = linesearch(x(:,k), p(:,k), fval(k), grad(:,k), alpha_0); % Determine alpha using Alg. 3.1
    x(:,k+1) = x(:,k) + alpha(k)*p(:,k);
    grad(:,k+1) = gradient(x(:,k+1)); 
    k = k+1;
    if k == maxiter
        disp('To many iteration is used')
    end
end
fval(k) = f(x(:,k)); % Final function value

% Delete unused space
x = x(:,1:k);
p = p(:,1:k);
grad = grad(:,1:k);
alpha = alpha(1:k);
fval = fval(1:k);

% Return values
x_opt = x(:,end);
fval_opt = f(x_opt);
x_iter = x;
f_iter = fval;

end

% Function returning the steepest-descent direction based on the gradien of f
function p = sd(grad)
    p = -grad;  % 'excpression for p';
end

% Function implementing Algorithm 3.1, page 37 in N&W
function alpha_k = linesearch(xk, pk, fk, gradk, alpha_0)
    alpha = alpha_0;
    rho = 0.95; %'contraction factor';
    c1 = 10^(-3); %'a constant for sufficient decrease';
    while f(xk + alpha*pk) > fk +c1*alpha*gradk'*pk  %'alpha is not good enough'
        alpha = rho*alpha; % 'a shorter step length';
    end
    alpha_k = alpha;  %'an alpha that is good enough';
end

% Function returning the value of the Rosenbrock function at x
function fval = f(x) 
    fval = 100*(x(2) - x(1)^2)^2 +(1-x(1))^2; %Rosenbrock function
end

% Function returning the value of the gradient at x
function grad = gradient(x)
    grad = [ (-400*x(1)*(x(2) - x(1)^2) - 2*(1 - x(1))) ; % Gradient to rosenbrock function
             (200*(x(2) - x(1)^2)) ];
end

function hes = hessian(x)
    hes = [ -400*x(2) + 1200*x(1)^2 + 2  ,  -400*x(1)  ; % Hessian to rosenbrock function
          -400*x(1)                    ,  200       ];
end