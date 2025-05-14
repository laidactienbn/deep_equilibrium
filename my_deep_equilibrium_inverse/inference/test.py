import torch
import numpy as np
from solvers import new_equilibrium_utils as eq_utils
from torch import autograd
from utils import cg_utils

def test_full_epoch(single_iterate_solver, test_dataloader,
                 measurement_process, optimizer,
                 save_location, loss_function, n_epochs, deep_eq_module,
                 use_dataparallel=False, device='cpu', scheduler=None, noise_sigma=0.000001, precond_iterates=100,
                 print_every_n_steps=2, save_every_n_epochs=5, start_epoch=0, forward_operator = None):
    previous_loss = 10.0
    reset_flag = False

    for epoch in range(start_epoch, n_epochs):

        if reset_flag:
            save_state_dict = torch.load(save_location)
            single_iterate_solver.load_state_dict(save_state_dict['solver_state_dict'])
            optimizer.load_state_dict(save_state_dict['optimizer_state_dict'])
        reset_flag = False

        for ii, sample_batch in enumerate(train_dataloader):
            print("Index:", ii)
            optimizer.zero_grad()

            sample_batch = sample_batch.to(device=device)
            y = measurement_process(sample_batch)
            if forward_operator is not None:
                with torch.no_grad():
                    initial_point = cg_utils.conjugate_gradient(initial_point=forward_operator.adjoint(y),
                                                                 ATA=forward_operator.gramian,
                                                                 regularization_lambda=noise_sigma, n_iterations=precond_iterates)
                reconstruction = deep_eq_module.forward(y, initial_point=initial_point)
            else:
                reconstruction = deep_eq_module.forward(y)
            loss = loss_function(reconstruction, sample_batch)
            if np.isnan(loss.item()):
                reset_flag = True
                break
            loss.backward()
            optimizer.step()

            if ii == 0:
                previous_loss = loss.item()

            if ii % print_every_n_steps == 0:
                logging_string = "Epoch: " + str(epoch) + " Step: " + str(ii) + \
                                 " Loss: " + str(loss.cpu().detach().numpy())
                print(logging_string, flush=True)

            if ii % 200 == 0:
                if use_dataparallel:
                    torch.save({'solver_state_dict': single_iterate_solver.module.state_dict(),
                                'epoch': epoch+1,
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': scheduler.state_dict()
                                }, save_location)
                else:
                    torch.save({'solver_state_dict': single_iterate_solver.state_dict(),
                                'epoch': epoch+1,
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': scheduler.state_dict()
                                }, save_location)

        if (previous_loss - loss.item()) / previous_loss < -10.0 or np.isnan(loss.item()):
            reset_flag = True

        if scheduler is not None:
            scheduler.step(epoch)
        if not reset_flag:
            if use_dataparallel:
                # torch.save({'solver_state_dict': single_iterate_solver.module.state_dict(),
                #             'epoch': epoch,
                #             'optimizer_state_dict': optimizer.state_dict(),
                #             'scheduler_state_dict': scheduler.state_dict()
                #             }, save_location + "_" + str(epoch))
                torch.save({'solver_state_dict': single_iterate_solver.module.state_dict(),
                            'epoch': epoch,
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict()
                            }, save_location)
            else:
                # torch.save({'solver_state_dict': single_iterate_solver.state_dict(),
                #             'epoch': epoch,
                #             'optimizer_state_dict': optimizer.state_dict(),
                #             'scheduler_state_dict': scheduler.state_dict()
                #             }, save_location + "_" + str(epoch))
                torch.save({'solver_state_dict': single_iterate_solver.state_dict(),
                            'epoch': epoch,
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict()
                            }, save_location)


def test_one_epoch(single_iterate_solver, test_dataloader,
                 measurement_process, optimizer,
                 save_location, loss_function, n_epochs, deep_eq_module,
                 use_dataparallel=False, device='cpu', scheduler=None, noise_sigma=0.000001, precond_iterates=100,
                 print_every_n_steps=2, save_every_n_epochs=5, start_epoch=0, forward_operator = None):
    previous_loss = 10.0
    # reset_flag = False


    # if reset_flag:
    #     save_state_dict = torch.load(save_location)
    #     single_iterate_solver.load_state_dict(save_state_dict['solver_state_dict'])
    #     optimizer.load_state_dict(save_state_dict['optimizer_state_dict'])
    # reset_flag = False
    single_iterate_solver.eval()
    scheduler.eval()
    for ii, sample_batch in enumerate(test_dataloader):
        print("Index:", ii)

        sample_batch = sample_batch.to(device=device)
        y = measurement_process(sample_batch)
        if forward_operator is not None:
            with torch.no_grad():
                initial_point = cg_utils.conjugate_gradient(initial_point=forward_operator.adjoint(y),
                                                                ATA=forward_operator.gramian,
                                                                regularization_lambda=noise_sigma, n_iterations=precond_iterates)
            reconstruction = deep_eq_module.forward(y, initial_point=initial_point)
        else:
            reconstruction = deep_eq_module.forward(y)
        loss = loss_function(reconstruction, sample_batch)
        if np.isnan(loss.item()):
            reset_flag = True
            break
        loss.backward()
        optimizer.step()

        if ii == 0:
            previous_loss = loss.item()

        if ii % print_every_n_steps == 0:
            logging_string = "Epoch: " + str(epoch) + " Step: " + str(ii) + \
                                " Loss: " + str(loss.cpu().detach().numpy())
            print(logging_string, flush=True)

        if ii % 200 == 0:
            if use_dataparallel:
                torch.save({'solver_state_dict': single_iterate_solver.module.state_dict(),
                            'epoch': epoch+1,
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict()
                            }, save_location)
            else:
                torch.save({'solver_state_dict': single_iterate_solver.state_dict(),
                            'epoch': epoch+1,
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict()
                            }, save_location)

        if (previous_loss - loss.item()) / previous_loss < -10.0 or np.isnan(loss.item()):
            reset_flag = True

        if scheduler is not None:
            scheduler.step(epoch)
        if not reset_flag:
            if use_dataparallel:
                # torch.save({'solver_state_dict': single_iterate_solver.module.state_dict(),
                #             'epoch': epoch,
                #             'optimizer_state_dict': optimizer.state_dict(),
                #             'scheduler_state_dict': scheduler.state_dict()
                #             }, save_location + "_" + str(epoch))
                torch.save({'solver_state_dict': single_iterate_solver.module.state_dict(),
                            'epoch': epoch,
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict()
                            }, save_location)
            else:
                # torch.save({'solver_state_dict': single_iterate_solver.state_dict(),
                #             'epoch': epoch,
                #             'optimizer_state_dict': optimizer.state_dict(),
                #             'scheduler_state_dict': scheduler.state_dict()
                #             }, save_location + "_" + str(epoch))
                torch.save({'solver_state_dict': single_iterate_solver.state_dict(),
                            'epoch': epoch,
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict()
                            }, save_location)


    #####################TEST##########################
    # loss_accumulator = []
    # mse_loss = torch.nn.MSELoss()
    # for ii, sample_batch in enumerate(test_dataloader):
    #     sample_batch = sample_batch.to(device=device)
    #     y = measurement_process(sample_batch)
    #     initial_point = y
    #     reconstruction = solver(initial_point, iterations=6)
    #
    #     reconstruction = torch.clamp(reconstruction, -1 ,1)
    #
    #     loss = mse_loss(reconstruction, sample_batch)
    #     loss_logger = loss.cpu().detach().numpy()
    #     loss_accumulator.append(loss_logger)
    #
    # loss_array = np.asarray(loss_accumulator)
    # loss_mse = np.mean(loss_array)
    # PSNR = -10 * np.log10(loss_mse)
    # percentiles = np.percentile(loss_array, [25,50,75])
    # percentiles = -10.0*np.log10(percentiles)
    # print("TEST LOSS: " + str(sum(loss_accumulator) / len(loss_accumulator)), flush=True)
    # print("MEAN TEST PSNR: " + str(PSNR), flush=True)
    # print("TEST PSNR QUARTILES AND MEDIAN: " + str(percentiles[0]) +
    #       ", " + str(percentiles[1]) + ", " + str(percentiles[2]), flush=True)
