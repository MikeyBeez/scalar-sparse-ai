/**
 * Core types for the Scalar-Sparse™ architecture
 */

/**
 * Scalar-Sparse™ token representation
 * Compresses traditional 4096-dimensional embeddings to <10 values
 */
export interface ScalarSparseToken {
  /** Core semantic scalar - the primary meaning value */
  baseValue: number;
  
  /** Learnable binary mask for routing computation (0 or 1 values) */
  sparseGates: number[];
  
  /** Contextual adjustment factor */
  modulator: number;
  
  /** Optional metadata for debugging/analysis */
  metadata?: {
    originalTokenId?: number;
    position?: number;
  };
}

/**
 * Token routing decision from the gating network
 */
export enum TokenRoute {
  /** Route to lightweight RNN for predictable patterns */
  Sequential = 'sequential',
  
  /** Route to MLP approximators for complex reasoning */
  DeepProcessing = 'deep'
}

/**
 * Configuration for the hybrid decoder
 */
export interface HybridDecoderConfig {
  /** Maximum context length in tokens */
  maxContextLength: number;
  
  /** Number of sparse gates per token */
  sparseGateCount: number;
  
  /** Threshold for routing decisions */
  routingThreshold: number;
  
  /** Number of specialized MLP approximators */
  mlpBankSize: number;
  
  /** Hidden size for lightweight RNN */
  rnnHiddenSize: number;
  
  /** Target dimension for teacher model reconstruction */
  teacherDimension: number;
}

/**
 * Compression statistics
 */
export interface CompressionStats {
  /** Original size in bytes */
  originalSize: number;
  
  /** Compressed size in bytes */
  compressedSize: number;
  
  /** Compression ratio (original / compressed) */
  compressionRatio: number;
  
  /** Context capacity with given memory */
  contextCapacity: {
    /** Number of tokens that fit in given memory */
    tokens: number;
    /** Memory size in GB */
    memoryGB: number;
  };
}
