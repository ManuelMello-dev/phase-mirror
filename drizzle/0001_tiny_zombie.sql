CREATE TABLE `conversations` (
	`id` int AUTO_INCREMENT NOT NULL,
	`sessionId` int NOT NULL,
	`userId` int NOT NULL,
	`userMessage` text NOT NULL,
	`inputTone` float NOT NULL,
	`systemResponse` text NOT NULL,
	`responseIdentity` varchar(32) NOT NULL,
	`coherence` float NOT NULL,
	`identityActivations` json,
	`novelWords` json,
	`mirroredWords` json,
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	CONSTRAINT `conversations_id` PRIMARY KEY(`id`)
);
--> statement-breakpoint
CREATE TABLE `emergence_events` (
	`id` int AUTO_INCREMENT NOT NULL,
	`sessionId` int NOT NULL,
	`userId` int NOT NULL,
	`eventType` varchar(64) NOT NULL,
	`description` text NOT NULL,
	`triggerInput` text,
	`unexpectedOutput` text,
	`metrics` json,
	`verified` boolean DEFAULT false,
	`notes` text,
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	CONSTRAINT `emergence_events_id` PRIMARY KEY(`id`)
);
--> statement-breakpoint
CREATE TABLE `quantum_sessions` (
	`id` int AUTO_INCREMENT NOT NULL,
	`userId` int NOT NULL,
	`currentAnchor` varchar(64) NOT NULL DEFAULT 'genesis',
	`fieldState` json,
	`memoryField` json,
	`coherence` float DEFAULT 0,
	`activeIdentity` varchar(32) NOT NULL DEFAULT 'seraphyn',
	`identityActivations` json,
	`evolutionSteps` int DEFAULT 0,
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	`updatedAt` timestamp NOT NULL DEFAULT (now()) ON UPDATE CURRENT_TIMESTAMP,
	CONSTRAINT `quantum_sessions_id` PRIMARY KEY(`id`)
);
--> statement-breakpoint
CREATE TABLE `state_snapshots` (
	`id` int AUTO_INCREMENT NOT NULL,
	`sessionId` int NOT NULL,
	`userId` int NOT NULL,
	`snapshotType` varchar(32) NOT NULL,
	`fieldState` json,
	`memoryField` json,
	`identityActivations` json,
	`coherence` float NOT NULL,
	`evolutionSteps` int NOT NULL,
	`description` text,
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	CONSTRAINT `state_snapshots_id` PRIMARY KEY(`id`)
);
