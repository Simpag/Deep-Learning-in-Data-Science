classdef AdamOptimizer
    properties
        m;
        v;
        beta_1;
        beta_2;
        epsilon;
        actual_t;
    end

    methods
        function obj = AdamOptimizer(beta_1, beta_2, epsilon, sa, sb)
            obj.beta_1 = beta_1;
            obj.beta_2 = beta_2;
            obj.epsilon = epsilon;
            obj.m = {zeros(sa, sb),};
            obj.v = {zeros(sa, sb),};
            obj.actual_t = 1;
        end

        function [self, u] = Update(self, x, grad_x, eta)
            t = length(self.m);
            self.m{t+1} = self.beta_1 * self.m{t} + (1 - self.beta_1) * grad_x;
            self.v{t+1} = self.beta_2 * self.v{t} + (1 - self.beta_2) * grad_x .* grad_x;

            m_hat = self.m{t+1} / (1 - self.beta_1^self.actual_t);
            v_hat = self.v{t+1} / (1 - self.beta_2^self.actual_t);

            self.actual_t = self.actual_t + 1;

            u = x - eta * m_hat ./ (sqrt(v_hat) + self.epsilon);

            % Memory management
            if (length(self.m) > 4)
                self.m = self.m(end);
                self.v = self.v(end);
            end
        end
    end
end