""" REWARD 1 - Possibly working

if p - f > 0:  ## Si la generación es mayor que la demanda
            if action_value < 0:  ## Si decido descargar la batería
                # print("CASO 1")
                reward = action_value*light_pvpc ## Acción errónea
            else:
                if (
                    battery_capacity >= self.maximum_battery_capacity
                ):  ## Si la batería está llena
                    
                    if action_value < 1e-4 and action_value > -1e-4: # This avoid probability 0 of event
                        # print("CASO 2")
                        reward = light_pvpc
                    else:
                        # print("CASO 3")
                        reward = -action_value*light_pvpc ## Penalización por almacenar
                else:
                    if action_value > p-f:
                        # print("Caso 4")
                        reward=(p-f)*light_pvpc + (p-f-action_value)*light_pvpc
                    else:
                        reward = action_value*light_pvpc
        elif p - f < 0:  ## Si la generación es menor que la demanda
            if action_value < 0:
                if battery_capacity!=0:
                    if action_value < -battery_capacity:
                        # print("CASO 7")
                        reward = (battery_capacity)*light_pvpc +action_value*light_pvpc ## Penalización por descargar demasiado, te premio por descargar algo
                    else:
                        if -action_value+p < f:
                            
                            if battery_capacity < f:
                                # print("CASO 8")
                                reward=-action_value*light_pvpc
                            else:
                                # print("CASO 9")
                                reward = -action_value*light_pvpc - (f+action_value-p)*light_pvpc
                        else:
                            # print("CASO 10")
                            # reward = -action_value*light_pvpc - (-action_value+p-f)*light_pvpc
                            reward=(f-p)*light_pvpc + (f-p+action_value)*light_pvpc
                else:
                    # print("CASO 11")
                    reward = action_value*light_pvpc
            else:
                # print("CASO 12")
                reward = -action_value*light_pvpc ## Penalización por almacenar
        else:
            if action_value > 1e-4:
                # print("CASO 13")
                reward = -action_value*light_pvpc ## Penalización por almacenar
            elif action_value < -1e-4:
                # print("CASO 14")
                reward = action_value*light_pvpc
            else:
                # print("CASO 15")
                reward = light_pvpc

"""