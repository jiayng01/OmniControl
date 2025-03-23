import torch


class DiscreteDPMSolver:
    """
    DPM-Solver implementation that works with discrete timesteps.
    Simplified to work with the ClassifierFreeSampleModel wrapper.
    """

    def __init__(self, model, noise_schedule, order=2):
        self.model = model
        self.alphas_cumprod = noise_schedule.alphas_cumprod
        self.total_steps = len(self.alphas_cumprod)
        self.order = order

        # Pre-compute noise schedule values
        self.sigma = torch.sqrt(1 - self.alphas_cumprod)
        self.alpha = torch.sqrt(self.alphas_cumprod)

    def get_model_output(self, x, t, model_kwargs):
        """
        Simply returns model output - guidance scaling is handled by wrapper.
        """
        return self.model(x, t, model_kwargs)

    def first_order_update(self, x, t_cur, t_next, model_kwargs):
        """
        First-order update step using discrete timesteps.
        """
        model_output = self.get_model_output(x, t_cur, model_kwargs)

        alpha_cur = self.alpha[t_cur]
        alpha_next = self.alpha[t_next]
        sigma_next = self.sigma[t_next]

        h = t_next - t_cur
        model_coefficient = sigma_next * h / (t_next - t_cur)

        x_next = (alpha_next / alpha_cur) * x - model_coefficient * model_output
        return x_next

    def second_order_update(self, x, t_cur, t_next, model_kwargs):
        """
        Second-order update step using discrete timesteps.
        """
        t_mid = (t_cur + t_next) // 2

        model_output_cur = self.get_model_output(x, t_cur, model_kwargs)
        x_mid = self.first_order_update(x, t_cur, t_mid, model_kwargs)
        model_output_mid = self.get_model_output(x_mid, t_mid, model_kwargs)

        alpha_cur = self.alpha[t_cur]
        alpha_next = self.alpha[t_next]
        sigma_next = self.sigma[t_next]

        h = t_next - t_cur
        h_mid = t_mid - t_cur

        model_output_diff = (model_output_mid - model_output_cur) * h / h_mid

        x_next = (alpha_next / alpha_cur) * x - (
            sigma_next * (model_output_cur + 0.5 * model_output_diff)
        )
        return x_next

    def sample(self, x, num_steps, model_kwargs=None):
        """
        Generates samples using DPM-Solver with discrete timesteps.
        """
        device = x.device
        batch_size = x.shape[0]

        steps = torch.linspace(
            self.total_steps - 1, 0, num_steps + 1, dtype=torch.long, device=device
        )

        for i in range(len(steps) - 1):
            t_cur = steps[i]
            t_next = steps[i + 1]

            t_cur_batch = torch.full((batch_size,), t_cur, device=device)
            t_next_batch = torch.full((batch_size,), t_next, device=device)

            # Choose update order
            if self.order == 1 or i == len(steps) - 2:
                x = self.first_order_update(x, t_cur_batch, t_next_batch, model_kwargs)
            else:
                x = self.second_order_update(x, t_cur_batch, t_next_batch, model_kwargs)

            # Apply spatial guidance if available
            if model_kwargs is not None and "y" in model_kwargs:
                if "hint" in model_kwargs["y"]:
                    if hasattr(self.model, "module"):
                        x = self.model.module.diffusion.guide(
                            x, t_next_batch, model_kwargs
                        )
                    else:
                        x = self.model.diffusion.guide(x, t_next_batch, model_kwargs)

        return x
