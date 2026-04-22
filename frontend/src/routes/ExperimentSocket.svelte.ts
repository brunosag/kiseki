import ReconnectingWebSocket from 'reconnecting-websocket';

export interface BaseExperimentConfig {
	device: 'gpu' | 'cpu';
	dataset: 'mnist' | 'fashion' | 'cifar10';
	seed: number;
	batchsize: number;
	max_i: number;
	target_acc: number;
}

export interface LEEAParams {
	type: 'leea';
	N: number;
	pₘ: number;
	'η₀': number;
	γ: number;
	ρ: number;
	ρₓ: number;
	λ: number;
	τ_pat: number;
}

export interface SGDParams {
	type: 'sgd';
	η: number;
}

export interface ExperimentConfig extends BaseExperimentConfig {
	opt: LEEAParams | SGDParams;
}

export interface OptimizerConfig {
	leea: LEEAParams;
	sgd: SGDParams;
}

export interface StepPayload {
	i: number;
	Δt: number;
	loss: number;
}

export interface ValidationPayload {
	i: number;
	value: number;
}

export interface SyncPayload {
	i: number;
	bestAcc: number;
	history: { loss: number[]; acc: { i: number; value: number }[] };
}

export interface StepMessage {
	type: 'step';
	payload: StepPayload;
}

export interface ValidationMessage {
	type: 'validation';
	payload: ValidationPayload;
}

export interface SyncMessage {
	type: 'sync';
	payload: SyncPayload;
}

export type Message = StepMessage | ValidationMessage | SyncMessage;

export class ExperimentSocket {
	isLoading = $state(false);
	i = $state(0);
	bestAcc = $state(Infinity);
	history = $state<{ loss: number[]; acc: { i: number; value: number }[] }>({
		loss: [],
		acc: []
	});
	isTraining = $state(false);

	#ws: ReconnectingWebSocket;

	constructor(url: string) {
		this.#ws = new ReconnectingWebSocket(url);

		this.#ws.onmessage = (event: MessageEvent) => {
			const msg = JSON.parse(event.data) as Message;
			switch (msg.type) {
				case 'sync':
					this.i = msg.payload.i;
					this.bestAcc = msg.payload.bestAcc;
					this.history = msg.payload.history;
					break;
				case 'step':
					this.i = msg.payload.i;
					this.history.loss.push(msg.payload.loss);
					break;
				case 'validation':
					this.history.acc.push({ i: msg.payload.i, value: msg.payload.acc });
					if (msg.payload.acc.acc > this.bestAcc) {
						this.bestAcc = msg.payload.acc.acc;
					}
					break;
			}
			this.isTraining = true;
			this.isLoading = false;
		};
	}

	startExperiment(config: ExperimentConfig) {
		this.isLoading = true;
		this.history = { loss: [], acc: [] };
		this.bestAcc = Infinity;
		this.#ws.send(JSON.stringify(config));
	}
}
