import { auth } from "@clerk/nextjs/server";
import { NextResponse } from "next/server";

import { db } from "@/lib/db";
import { DocumentStatus } from "@/types";
import { processDocument } from "@/lib/rag";

export async function POST(
  req: Request,
  { params }: { params: { documentId: string } }
) {
  try {
    const { userId } = await auth();

    if (!userId) {
      return new NextResponse("Unauthorized", { status: 401 });
    }

    // Find the document and check if user has access
    const document = await db.document.findFirst({
      where: {
        id: params.documentId,
        workspace: {
          members: {
            some: {
              user: {
                clerkId: userId
              }
            }
          }
        }
      }
    });

    if (!document) {
      return new NextResponse("Document not found", { status: 404 });
    }

    // Update status to PROCESSING
    await db.document.update({
      where: { id: params.documentId },
      data: { 
        status: DocumentStatus.PROCESSING,
        error: null
      }
    });

    // Process document with RAG service
    try {
      const result = await processDocument({
        documentId: document.id,
        fileUrl: document.url,
        workspaceId: document.workspaceId,
        fileName: document.name,
        fileType: document.type as any // TODO: Fix type casting
      });

      // Update status to COMPLETED and save vectorIds
      await db.document.update({
        where: { id: params.documentId },
        data: { 
          status: DocumentStatus.COMPLETED,
          vectorIds: result.vectorIds
        }
      });
    } catch (error) {
      // Update status to FAILED
      await db.document.update({
        where: { id: params.documentId },
        data: { 
          status: DocumentStatus.FAILED,
          error: error instanceof Error ? error.message : "Unknown error"
        }
      });
      throw error;
    }

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error("[DOCUMENT_PROCESS_RETRY]", error);
    return new NextResponse("Internal Error", { status: 500 });
  }
} 